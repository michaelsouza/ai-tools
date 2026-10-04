# pt-BR TTS benchmark

Compares text-to-speech engines on Brazilian Portuguese, to pick a local engine
that fits the 8 GB RTX 4070 Laptop and a cheap paid fallback.

## Method

- **Corpus:** [sentences.json](sentences.json), 20 sentences covering everyday
  speech, nasal vowels, numbers and money, Brazilian place and author names,
  acronyms, abbreviations, academic prose, English loanwords, questions and
  exclamations, a tongue twister, regional vocabulary and two long sentences.
- **Voice:** zero-shot cloning of one 15.2 s clip of a female Brazilian
  narrator ([prepare.py](prepare.py); dataset `alissonpadua/tts-dataset-pt-br`,
  MIT). Engines that cannot clone use their own voice: Voxtral `pt_female`
  (the open weights ship presets only), Gemini `Kore`.
- **Intelligibility:** [evaluate.py](evaluate.py) transcribes each clip with
  faster-whisper large-v3 and reports WER/CER after normalization (numbers
  spelled out, punctuation dropped). *Core* WER leaves out s04, s05 and s09,
  whose numbers and abbreviations have several valid readings. Control: Whisper
  on the reference clip scores 0% WER.
- **Speed and memory:** RTF = generation time / audio duration (warm model,
  one sentence at a time) and peak `nvidia-smi` memory. For the APIs the time
  is the HTTP round trip.
- **Listening:** [make_listening.py](make_listening.py) builds a blind
  A–E page (1–5 ratings); the published copy saves ratings per viewer.

## Results (2026-10-04)

| engine | where | voice | WER core | CER core | WER all | RTF | peak VRAM |
|---|---|---|---|---|---|---|---|
| Fish S2.1 Pro | OpenRouter API | clone | 1.1% | 0.2% | 2.1% | 0.35 | — |
| Qwen3-TTS 1.7B Base, bf16 | local | clone | 1.4% | 0.3% | 3.7% | 1.77 | 5.5 GB |
| Fish S2 Pro Q6_K (s2.cpp) | local | clone | 1.7% | 0.4% | 3.7% | 2.02 | 7.6 GB |
| Gemini 3.8 Flash-Lite TTS | OpenRouter API | Kore | 2.2% | 0.6% | 3.7% | 0.52 | — |
| Voxtral 4B TTS int4 (voxtral-int4) | local | pt_female | 11.0% | 7.9% | 18.6% | 1.78 | 7.7 GB |

- Voxtral int4 garbles sentence openings and hallucinates whole clauses
  (s04 came out as unrelated speech); a second seed moved the errors around
  without fixing them (core WER 12.7% on the six worst sentences). The
  full-precision model needs 16 GB and the OpenRouter endpoint has no
  Portuguese voice, so the cause (quantization vs. model) is untested.
- Only Gemini read the number and abbreviation sentences (s04, s05, s09)
  correctly according to Whisper. Qwen3 and both Fish runs garble
  `R$ 1.250.000,00`; Fish (local and API) drops the "30" of `14h30` and
  reads "sala 204" differently; Qwen3 and Fish read `Profa.` literally.
  Whisper's own number formatting adds ambiguity, so confirm by ear.
- Fish S2 Pro slows down on long sentences (RTF 2.8 on s19/s20).
- Cost of the API runs: about 2,700 characters per engine, a few US cents.

### Blind listening (one rater, 1–5, [results/listening_ratings.json](results/listening_ratings.json))

| engine | mean | sd | 5s | ≤2s |
|---|---|---|---|---|
| Gemini 3.8 Flash-Lite TTS | 4.95 | 0.22 | 19 | 0 |
| Fish S2.1 Pro (API) | 3.50 | 0.83 | 2 | 2 |
| Fish S2 Pro Q6_K (local) | 3.40 | 0.99 | 3 | 4 |
| Qwen3-TTS 1.7B (local) | 3.25 | 0.91 | 2 | 4 |
| Voxtral 4B int4 (local) | 2.85 | 1.18 | 0 | 9 |

Gemini won 19 of 20 sentences. Fish local beat Qwen3 on 8 sentences and
lost on 4 (sign test p = 0.39, no real difference). Gemini's own studio voice
was recognizable among the cloned ones, so the test was not fully blind for
it, and the cloned engines inherit the quality of a 22 kHz audiobook
reference.

## Running

```bash
python benchmarks/tts/prepare.py
python benchmarks/tts/run_qwen.py                     # main venv (qwen-tts)
python benchmarks/tts/run_s2.py --quant q6_k          # needs ~/gitrepos/s2.cpp/build/s2 (CUDA)
~/gitrepos/voxtral-int4/venv/bin/python benchmarks/tts/run_voxtral.py
python benchmarks/tts/run_openrouter.py gemini-flash-lite
python benchmarks/tts/run_openrouter.py fish-s2.1-api
python benchmarks/tts/evaluate.py
python benchmarks/tts/make_listening.py
```

Models live in `models/tts-bench/` (Qwen/Qwen3-TTS-12Hz-1.7B-Base,
mistralai/Voxtral-4B-TTS-2603, rodrigomt/s2-pro-gguf). Build s2.cpp with
`cmake -B build -DS2_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`. voxtral-int4
needs its own venv (torch ≥ 2.11, torchao, hqq, tiktoken, scipy); its
`generate_speech_fast` returns 48 kHz audio, although its `run.sh` saves 24 kHz.
Run one local engine at a time and nothing CPU-heavy alongside: a parallel
CUDA build made Qwen3 25× slower.

## Limitations

- One voice, 20 sentences, one sample per sentence (seed 0).
- WER measures intelligibility, not naturalness or accent; that is what the
  listening page is for.
- Voxtral and Gemini do not clone, so their voice differs from the others.
