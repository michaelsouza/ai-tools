#!/usr/bin/env python3
"""
Generates a flowchart from Python, C, or C++ source files by analyzing function calls.
The default output is a PNG image. JSON and SVG outputs are optional.
"""

import ast
import argparse
import sys
from pathlib import Path
import subprocess
from datetime import datetime
import json

try:
    from tree_sitter import Language, Parser
    import tree_sitter_c as ts_c
    import tree_sitter_cpp as ts_cpp

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


class FunctionCallVisitor(ast.NodeVisitor):
    """
    An AST visitor to trace function calls and build a call graph.
    """

    def __init__(self):
        self.graph = {}
        self.current_function = None
        self.function_stack = []

    def visit_FunctionDef(self, node):
        """Visit a function definition."""
        self.function_stack.append(node.name)
        self.current_function = node.name
        if self.current_function not in self.graph:
            self.graph[self.current_function] = []

        self.generic_visit(node)

        self.function_stack.pop()
        if self.function_stack:
            self.current_function = self.function_stack[-1]
        else:
            self.current_function = None

    def visit_Call(self, node):
        """Visit a function call."""
        if self.current_function:
            callee_name = self.get_callee_name(node.func)
            if callee_name:
                self.graph[self.current_function].append(callee_name)

                if callee_name.endswith("add_node") and len(node.args) > 1:
                    node_arg = node.args[1]
                    if isinstance(node_arg, ast.Name):
                        self.graph[self.current_function].append(node_arg.id)
                    elif isinstance(node_arg, ast.Lambda) and isinstance(node_arg.body, ast.Call):
                        lambda_callee = self.get_callee_name(node_arg.body.func)
                        if lambda_callee:
                            self.graph[self.current_function].append(lambda_callee)

                if callee_name.endswith("add_conditional_edges") and len(node.args) > 1:
                    cond_arg = node.args[1]
                    if isinstance(cond_arg, ast.Name):
                        self.graph[self.current_function].append(cond_arg.id)

        self.generic_visit(node)

    def get_callee_name(self, func_node):
        """Recursively get the name of a callee."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            return func_node.attr
        return None


def analyze_code(file_path):
    """
    Reads and analyzes Python code to build a function call graph and get defined functions.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code)

    defined_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    visitor = FunctionCallVisitor()
    visitor.visit(tree)

    return visitor.graph, defined_functions


def filter_graph(graph, defined_functions):
    """Filters a call graph to only include functions defined in the source file."""
    filtered_graph = {func: [] for func in defined_functions}

    for caller, callees in graph.items():
        if caller in defined_functions:
            filtered_callees = [callee for callee in callees if callee in defined_functions]
            if filtered_callees:
                filtered_graph[caller].extend(filtered_callees)
    return filtered_graph


class CCppAnalyzer:
    """
    Analyzer for C/C++ code using tree-sitter to build a function call graph.
    """

    def __init__(self, language="c"):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter packages not available. Install with: pip install tree-sitter tree-sitter-c tree-sitter-cpp"
            )

        self.language = language
        self.parser = Parser()
        if language == "cpp":
            self.parser.set_language(Language(ts_cpp.language(), "cpp"))
        else:
            self.parser.set_language(Language(ts_c.language(), "c"))

        self.graph = {}
        self.defined_functions = set()

    def analyze(self, code):
        """Analyzes C/C++ code and builds a call graph."""
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        # Find all function definitions
        self._find_function_definitions(root_node)

        # Build call graph
        self._build_call_graph(root_node)

        return self.graph, self.defined_functions

    def _find_function_definitions(self, node):
        """Recursively find all function definitions."""
        if node.type == "function_definition":
            func_name = self._get_function_name(node)
            if func_name:
                self.defined_functions.add(func_name)
                if func_name not in self.graph:
                    self.graph[func_name] = []

        for child in node.children:
            self._find_function_definitions(child)

    def _get_function_name(self, func_def_node):
        """Extract function name from a function_definition node."""
        declarator = func_def_node.child_by_field_name("declarator")
        if not declarator:
            return None

        # Handle different declarator types
        while declarator:
            if declarator.type == "function_declarator":
                declarator = declarator.child_by_field_name("declarator")
            elif declarator.type == "pointer_declarator":
                declarator = declarator.child_by_field_name("declarator")
            elif declarator.type == "identifier":
                return declarator.text.decode("utf8")
            else:
                # Try to find identifier in children
                for child in declarator.children:
                    if child.type == "identifier":
                        return child.text.decode("utf8")
                break

        return None

    def _build_call_graph(self, node):
        """Build the call graph by finding function calls within each function."""
        if node.type == "function_definition":
            func_name = self._get_function_name(node)
            if func_name:
                self._find_calls_in_function(node, func_name)

        for child in node.children:
            self._build_call_graph(child)

    def _find_calls_in_function(self, func_node, caller_name):
        """Find all function calls within a function."""
        body = func_node.child_by_field_name("body")
        if body:
            self._extract_calls(body, caller_name)

    def _extract_calls(self, node, caller_name):
        """Recursively extract function calls from a node."""
        if node.type == "call_expression":
            function_node = node.child_by_field_name("function")
            if function_node:
                callee_name = self._get_call_name(function_node)
                if callee_name and caller_name in self.graph:
                    self.graph[caller_name].append(callee_name)

        for child in node.children:
            self._extract_calls(child, caller_name)

    def _get_call_name(self, func_node):
        """Extract the name of a called function."""
        if func_node.type == "identifier":
            return func_node.text.decode("utf8")
        elif func_node.type == "field_expression":
            field = func_node.child_by_field_name("field")
            if field and field.type == "field_identifier":
                return field.text.decode("utf8")
        elif func_node.type == "qualified_identifier" or func_node.type == "scoped_identifier":
            # For C++ qualified names like std::cout
            name = func_node.child_by_field_name("name")
            if name:
                return name.text.decode("utf8")

        return None


def analyze_c_cpp_code(file_path, language="c"):
    """
    Reads and analyzes C/C++ code to build a function call graph.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    analyzer = CCppAnalyzer(language)
    graph, defined_functions = analyzer.analyze(code)

    return graph, defined_functions


def detect_language(file_path):
    """
    Detects the programming language based on file extension.
    Returns 'python', 'c', or 'cpp'.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".py":
        return "python"
    elif suffix in [".c", ".h"]:
        return "c"
    elif suffix in [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"]:
        return "cpp"
    else:
        return None


def generate_dot_graph(graph, script_name):
    """
    Generates a Graphviz DOT representation of the call graph.
    """
    dot_lines = ["digraph G {"]
    dot_lines.append(f'  label="{script_name}";')
    dot_lines.append("  labelloc=t;")
    dot_lines.append("  rankdir=LR;")
    dot_lines.append('  node [shape=box, style=rounded, fontname="Helvetica"];')
    dot_lines.append('  edge [fontname="Helvetica"];')

    all_nodes = set(graph.keys())
    for callees in graph.values():
        all_nodes.update(callees)

    for node in sorted(list(all_nodes)):
        dot_lines.append(f'  "{node}";')

    for caller, callees in graph.items():
        if callees:
            for callee in sorted(list(set(callees))):
                dot_lines.append(f'  "{caller}" -> "{callee}";')

    dot_lines.append("}")
    return "\n".join(dot_lines)


def main():
    """
    Main function to parse arguments and run the analysis.
    """
    parser = argparse.ArgumentParser(
        description="Generate a flowchart PNG from Python, C, or C++ source files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("source_file", help="Path to the source file to analyze (Python, C, or C++).")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not generate PNG image (requires Graphviz).",
    )
    parser.add_argument("--json", action="store_true", help="Also generate JSON output file.")
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also generate SVG image file (requires Graphviz).",
    )
    parser.add_argument(
        "--print-dot",
        action="store_true",
        help="Print the DOT representation to stdout after generating files.",
    )
    args = parser.parse_args()

    script_path = Path(args.source_file)

    # Detect language
    language = detect_language(script_path)
    if language is None:
        print(
            f"Error: Unsupported file extension '{script_path.suffix}'. Supported: .py, .c, .h, .cpp, .cc, .cxx, .hpp, .hh, .hxx",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Analyze code based on language
        if language == "python":
            full_graph, defined_functions = analyze_code(script_path)
        elif language in ["c", "cpp"]:
            if not TREE_SITTER_AVAILABLE:
                print(
                    "Error: tree-sitter packages are required for C/C++ analysis.",
                    file=sys.stderr,
                )
                print(
                    "Install with: pip install tree-sitter tree-sitter-c tree-sitter-cpp",
                    file=sys.stderr,
                )
                sys.exit(1)
            full_graph, defined_functions = analyze_c_cpp_code(script_path, language)
        else:
            print(f"Error: Unsupported language '{language}'", file=sys.stderr)
            sys.exit(1)
        graph = filter_graph(full_graph, defined_functions)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_filename = f"{script_path.stem}_flowchart_{timestamp}"

        # Generate JSON only if requested
        if args.json:
            json_output_path = script_path.parent / f"{base_filename}.json"
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2, ensure_ascii=False)
            print(f"Call graph data saved to: {json_output_path}")

        dot_representation = generate_dot_graph(graph, script_path.name)

        if args.print_dot:
            print("\n--- DOT Representation ---")
            print(dot_representation)

        if args.no_images:
            return

        # --- Image Generation ---
        png_output_path = script_path.parent / f"{base_filename}.png"

        try:
            # Generate PNG
            subprocess.run(
                ["dot", "-Tpng", "-o", str(png_output_path)],
                input=dot_representation,
                encoding="utf-8",
                check=True,
                capture_output=True,
            )
            print(f"Flowchart image saved to: {png_output_path}")

            # Generate SVG only if requested
            if args.svg:
                svg_output_path = script_path.parent / f"{base_filename}.svg"
                subprocess.run(
                    ["dot", "-Tsvg", "-o", str(svg_output_path)],
                    input=dot_representation,
                    encoding="utf-8",
                    check=True,
                    capture_output=True,
                )
                print(f"Flowchart image saved to: {svg_output_path}")

        except FileNotFoundError:
            print(
                "\nWarning: 'dot' command not found (Graphviz). Images were not generated.",
                file=sys.stderr,
            )
        except subprocess.CalledProcessError as e:
            print("\nError executing 'dot' command:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Input file not found at {script_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
