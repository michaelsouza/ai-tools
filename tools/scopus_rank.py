#!/home/michael/gitrepos/ai-tools/.venv/bin/python
"""
Scopus Reference Ranker (Jev / OpenRouter)

Scores and ranks the references of a Scopus CSV export against a research
topic using Jev, TypeSafe AI's decision model, through OpenRouter.

Jev does not generate text. For each reference it receives a state (topic,
title, abstract, keywords...) plus the typed questions of a screening profile
and returns typed answers with probabilities. One request is sent per
reference; all questions of the profile are answered in that single request.

Usage:
    python scopus_rank.py scopus.csv --profile profile.json [options]

Profile (JSON):
    {
      "topic": "free-text description of the research",
      "rank_by": "relevance",
      "questions": {
        "relevance": {"type": "score", "instructions": "...", "criteria": ["lowest level", "...", "highest level"]},
        "on_topic": {"type": "noul", "instructions": "..."},
        "kind": {"type": "choice", "instructions": "...", "criteria": {"label": "description", ...}}
      }
    }
    `score` criteria are an ordered list of levels (worst to best); the answer
    is a continuous value in [0, levels - 1]. `noul` answers are P(yes).
    `choice` criteria map each label to its description.
    `rank_by` must name a `score` or `noul` question.
    See docs/scopus_rank_profile.example.json.

Options:
    -o, --output PATH   Ranked CSV (default: <input stem>.ranked.csv)
    --profile PATH      Screening profile (required)
    --model ID          OpenRouter model id (default: ~typesafe/jev-latest)
    --workers N         Concurrent requests (default: 8)
    --limit N           Only process the first N references (profile calibration)
    --dry-run           Print one payload and a token/cost estimate; no network
    --no-cache          Ignore and do not write the <input stem>.jev.jsonl cache

Output:
    The original CSV plus one `jev_<question>` column per question
    (`jev_<question>_conf` / `jev_<question>_p` when available) and `jev_rank`,
    sorted by the `rank_by` question (descending), ties broken by `Cited by`.
    Answers are cached in <input stem>.jev.jsonl; re-running with the same
    profile and model only requests the missing references.

Requirements:
    OPENROUTER_API_KEY in the repo `.env` (Jev has no free tier).

Exit Codes:
    0 - Success
    1 - Processing error (some references could not be scored)
    2 - Configuration error (bad arguments, profile, CSV or missing API key)
"""

import argparse
import csv
import hashlib
import json
import os
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

try:
    from tools.merge_bib import normalize_doi, normalize_title
except ImportError:
    from merge_bib import normalize_doi, normalize_title

JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"
USD_PER_MILLION_INPUT_TOKENS = 0.042
QUESTION_TYPES = ("score", "noul", "choice")
NO_ABSTRACT_MARKERS = ("", "[no abstract available]")
MAX_ATTEMPTS = 4


class JevError(Exception):
    pass


def load_scopus_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Reads a Scopus export. Returns (rows, fieldnames)."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def record_id(row: Dict[str, str]) -> str:
    """Stable identifier of a reference: EID, then DOI, then normalized title."""
    eid = (row.get("EID") or "").strip()
    if eid:
        return f"eid:{eid}"
    doi = normalize_doi(row.get("DOI") or "")
    if doi:
        return f"doi:{doi}"
    return f"title:{normalize_title(row.get('Title') or '')}"


def load_profile(path: Path) -> Tuple[Optional[dict], List[str]]:
    """Loads and validates a screening profile. Returns (profile, errors)."""
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path}: {exc}"]

    errors = []
    if not isinstance(profile, dict):
        return None, [f"{path}: profile must be a JSON object"]
    if not str(profile.get("topic", "")).strip():
        errors.append("profile: 'topic' is required")
    questions = profile.get("questions")
    if not isinstance(questions, dict) or not questions:
        errors.append("profile: 'questions' must be a non-empty object")
        return None, errors
    for qid, question in questions.items():
        qtype = question.get("type") if isinstance(question, dict) else None
        if qtype not in QUESTION_TYPES:
            errors.append(f"profile: question '{qid}' must have type in {QUESTION_TYPES}")
        elif not str(question.get("instructions", "")).strip():
            errors.append(f"profile: question '{qid}' needs 'instructions'")
        elif qtype == "score" and not (isinstance(question.get("criteria"), list) and len(question["criteria"]) >= 2):
            errors.append(f"profile: question '{qid}' (score) needs 'criteria' as a list of ordered levels")
        elif qtype == "choice" and not (isinstance(question.get("criteria"), dict) and question["criteria"]):
            errors.append(f"profile: question '{qid}' (choice) needs 'criteria' as an object of label: description")
    rank_by = profile.get("rank_by")
    if rank_by not in questions:
        errors.append("profile: 'rank_by' must name one of the questions")
    elif isinstance(questions[rank_by], dict) and questions[rank_by].get("type") == "choice":
        errors.append("profile: 'rank_by' must be a score or noul question")
    return (None, errors) if errors else (profile, [])


def profile_hash(profile: dict, model: str) -> str:
    blob = json.dumps(
        {"topic": profile["topic"], "questions": profile["questions"], "model": model},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_state(row: Dict[str, str], topic: str, max_abstract_chars: Optional[int] = None) -> dict:
    """Builds the Jev state for one reference."""
    abstract = (row.get("Abstract") or "").strip()
    has_abstract = abstract.casefold() not in NO_ABSTRACT_MARKERS
    if has_abstract and max_abstract_chars:
        abstract = abstract[:max_abstract_chars]
    keywords = "; ".join(
        value.strip() for value in (row.get("Author Keywords"), row.get("Index Keywords")) if value and value.strip()
    )
    state = {
        "research_topic": topic,
        "title": (row.get("Title") or "").strip(),
        "abstract": abstract if has_abstract else "",
        "no_abstract": not has_abstract,
        "keywords": keywords,
        "year": (row.get("Year") or "").strip(),
        "source": (row.get("Source title") or "").strip(),
        "document_type": (row.get("Document Type") or "").strip(),
    }
    return state


def build_decision_payload(state: dict, questions: dict, model: str) -> dict:
    """Builds the OpenRouter decisions request payload."""
    return {"model": model, "state": state, "questions": questions}


def _first_number(answer: dict, keys: Tuple[str, ...]) -> Optional[float]:
    for key in keys:
        value = answer.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _flatten_answer(qid: str, qtype: str, answer: Any) -> Dict[str, Any]:
    column = f"jev_{qid}"
    if not isinstance(answer, dict):
        return {column: answer}

    probabilities = answer.get("probabilities")
    probabilities = probabilities if isinstance(probabilities, dict) else {}
    flat: Dict[str, Any] = {}

    if qtype == "choice":
        label = answer.get("choice") or answer.get("answer") or answer.get("value")
        if label is None and probabilities:
            label = max(probabilities, key=probabilities.get)
        flat[column] = label
        if label in probabilities:
            flat[f"{column}_p"] = probabilities[label]
    elif qtype == "noul":
        value = _first_number(answer, ("noul", "probability"))
        flat[column] = value
    else:
        flat[column] = _first_number(answer, ("score",))

    confidence = answer.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        flat[f"{column}_conf"] = float(confidence)
    return flat


def parse_answers(response: dict, questions: dict) -> Dict[str, Any]:
    """Flattens a decisions response into `jev_<question>` columns."""
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        raise JevError(f"response has no 'answers': {str(response)[:200]}")
    flat: Dict[str, Any] = {}
    for qid, question in questions.items():
        if qid in answers:
            flat.update(_flatten_answer(qid, question["type"], answers[qid]))
    return flat


def _create_https_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def call_jev(payload: dict, api_key: str) -> dict:
    """POSTs one decisions request, retrying on 429/5xx. Raises JevError."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urlrequest.Request(
            JEV_ENDPOINT,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=60, context=_create_https_context()) as response:
                return json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 413:
                raise JevError("HTTP 413: payload too large") from exc
            if exc.code != 429 and exc.code < 500 or attempt == MAX_ATTEMPTS:
                raise JevError(f"HTTP {exc.code}: {detail}") from exc
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise JevError(str(exc)) from exc
        time.sleep(2**attempt)
    raise JevError("unreachable")


def score_reference(row: Dict[str, str], profile: dict, model: str, api_key: str) -> dict:
    """Scores one reference. Returns a cache entry (without id/profile_hash)."""
    questions = profile["questions"]
    try:
        response = call_jev(build_decision_payload(build_state(row, profile["topic"]), questions, model), api_key)
    except JevError as exc:
        if "413" not in str(exc):
            raise
        state = build_state(row, profile["topic"], max_abstract_chars=4000)
        response = call_jev(build_decision_payload(state, questions, model), api_key)
    return {"answers": parse_answers(response, questions), "usage": response.get("usage") or {}}


def load_cache(path: Path, wanted_hash: str) -> Dict[str, dict]:
    """Loads cache entries matching the profile hash, keyed by record id."""
    entries: Dict[str, dict] = {}
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("profile_hash") == wanted_hash and "id" in entry:
                entries[entry["id"]] = entry
    return entries


def _cited_by(row: Dict[str, Any]) -> int:
    try:
        return int(str(row.get("Cited by") or "0").strip() or 0)
    except ValueError:
        return 0


def rank_rows(rows: List[Dict[str, Any]], rank_column: str) -> List[Dict[str, Any]]:
    """Sorts by rank column (desc), then citations; unscored rows go last. Sets jev_rank."""

    def sort_key(row: Dict[str, Any]):
        value = row.get(rank_column)
        scored = isinstance(value, (int, float))
        return (0 if scored else 1, -(value if scored else 0), -_cited_by(row))

    ranked = sorted(rows, key=sort_key)
    for position, row in enumerate(ranked, start=1):
        row["jev_rank"] = position
    return ranked


def write_ranked_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    extra: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("jev_") and key != "jev_rank" and key not in extra:
                extra.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["jev_rank"] + extra + fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except Exception:
        return len(text) // 4


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank a Scopus CSV export against a research topic using Jev.")
    parser.add_argument("csv", type=Path, help="Scopus CSV export")
    parser.add_argument("--profile", type=Path, required=True, help="Screening profile (JSON)")
    parser.add_argument("-o", "--output", type=Path, help="Ranked CSV (default: <stem>.ranked.csv)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests (default: 8)")
    parser.add_argument("--limit", type=int, help="Only process the first N references")
    parser.add_argument("--dry-run", action="store_true", help="Print one payload and a cost estimate; no network")
    parser.add_argument("--no-cache", action="store_true", help="Ignore and do not write the .jev.jsonl cache")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> List[str]:
    errors = []
    if not args.csv.is_file():
        errors.append(f"CSV not found: {args.csv}")
    if not args.profile.is_file():
        errors.append(f"Profile not found: {args.profile}")
    if args.workers < 1:
        errors.append("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        errors.append("--limit must be >= 1")
    return errors


def print_dry_run(console: Console, rows: List[Dict[str, str]], profile: dict, model: str) -> None:
    payloads = [build_decision_payload(build_state(row, profile["topic"]), profile["questions"], model) for row in rows]
    console.print_json(json.dumps(payloads[0], ensure_ascii=False))
    tokens = sum(estimate_tokens(json.dumps(payload, ensure_ascii=False)) for payload in payloads)
    cost = tokens / 1_000_000 * USD_PER_MILLION_INPUT_TOKENS
    console.print(f"\n{len(rows)} references, ~{tokens:,} input tokens, ~US$ {cost:.4f} (estimate)")


def print_summary(console: Console, ranked: List[Dict[str, Any]], rank_column: str, usage: Dict[str, float]) -> None:
    table = Table(title=f"Top references by {rank_column}")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column(rank_column.removeprefix("jev_"), justify="right", no_wrap=True, min_width=6)
    table.add_column("Cited", justify="right", min_width=5)
    table.add_column("Year", min_width=4)
    table.add_column("Title")
    for row in ranked[:15]:
        value = row.get(rank_column)
        title = row.get("Title") or ""
        table.add_row(
            str(row["jev_rank"]),
            f"{value:.3f}" if isinstance(value, (int, float)) else "-",
            str(_cited_by(row)),
            row.get("Year") or "",
            title if len(title) <= 60 else title[:59] + "…",
        )
    console.print(table)

    values = [row[rank_column] for row in ranked if isinstance(row.get(rank_column), (int, float))]
    if values:
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        bins = [0] * 5
        for value in values:
            bins[min(4, int((value - low) / span * 5))] += 1
        parts = [f"[{low + span * i / 5:.2f}-{low + span * (i + 1) / 5:.2f}]: {count}" for i, count in enumerate(bins)]
        console.print("Distribution  " + "  ".join(parts))
    console.print(f"Requests: {int(usage['requests'])}  input tokens: {int(usage['tokens']):,}  cost: US$ {usage['cost']:.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    console = Console()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    args = parse_args(argv)

    errors = validate_args(args)
    profile = None
    if not errors:
        profile, errors = load_profile(args.profile)
    if errors or profile is None:
        for message in errors:
            console.print(f"[bold red]Error:[/] {message}")
        return 2

    rows, fieldnames = load_scopus_csv(args.csv)
    if "Title" not in fieldnames:
        console.print(f"[bold red]Error:[/] {args.csv} has no 'Title' column; is it a Scopus CSV export?")
        return 2
    if "Abstract" not in fieldnames:
        console.print("[yellow]Warning:[/] no 'Abstract' column; export it from Scopus for meaningful scores.")
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        console.print("[bold red]Error:[/] no references in CSV.")
        return 2

    if args.dry_run:
        print_dry_run(console, rows, profile, args.model)
        return 0

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[bold red]Error:[/] OPENROUTER_API_KEY not set (repo .env).")
        return 2

    wanted_hash = profile_hash(profile, args.model)
    cache_path = args.csv.with_suffix(".jev.jsonl")
    cache = {} if args.no_cache else load_cache(cache_path, wanted_hash)

    pending: Dict[str, Dict[str, str]] = {}
    for row in rows:
        rid = record_id(row)
        if rid not in cache:
            pending.setdefault(rid, row)
    console.print(f"{len(rows)} references, {len(rows) - len(pending)} cached, {len(pending)} to score.")

    usage = {"requests": 0.0, "tokens": 0.0, "cost": 0.0}
    failures: List[str] = []
    cache_handle = None if args.no_cache else open(cache_path, "a", encoding="utf-8")
    try:
        with Progress(
            SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(), console=console,
        ) as progress, ThreadPoolExecutor(max_workers=args.workers) as pool:
            task = progress.add_task("Scoring with Jev", total=len(pending))
            futures = {pool.submit(score_reference, row, profile, args.model, api_key): rid for rid, row in pending.items()}
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    entry = {"id": rid, "profile_hash": wanted_hash, **future.result()}
                except Exception as exc:
                    failures.append(f"{rid}: {exc}")
                else:
                    cache[rid] = entry
                    usage["requests"] += 1
                    usage["tokens"] += entry["usage"].get("input_tokens") or 0
                    usage["cost"] += entry["usage"].get("cost") or 0
                    if cache_handle:
                        cache_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        cache_handle.flush()
                progress.advance(task)
    finally:
        if cache_handle:
            cache_handle.close()

    for row in rows:
        entry = cache.get(record_id(row))
        if entry:
            row.update(entry["answers"])

    rank_column = f"jev_{profile['rank_by']}"
    ranked = rank_rows(rows, rank_column)
    output = args.output or args.csv.with_suffix(".ranked.csv")
    write_ranked_csv(output, ranked, fieldnames)

    print_summary(console, ranked, rank_column, usage)
    console.print(f"[green]Ranked CSV:[/] {output}")
    if failures:
        console.print(f"[bold red]{len(failures)} references failed[/] (re-run to retry):")
        for message in failures[:10]:
            console.print(f"  {message}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
