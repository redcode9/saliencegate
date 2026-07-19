from __future__ import annotations

import ast
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRECTORY = Path("src/saliencegate/benchmarks/state_decay_v2")
BOUNDARY_MODULES = (
    "public_contract.py",
    "generation_authority.py",
    "public_catalog.py",
    "templates.py",
    "preallocation.py",
    "signal_fixtures.py",
    "review_contract.py",
    "review.py",
    "review_pack.py",
    "review_io.py",
    "review_cli.py",
)

_NETWORK_MODULES = frozenset(
    {
        "_socket",
        "aiohttp",
        "ftplib",
        "grpc",
        "http.client",
        "http.server",
        "httpx",
        "httpcore",
        "httplib2",
        "pycurl",
        "requests",
        "smtplib",
        "socket",
        "telnetlib",
        "urllib.request",
        "urllib3",
        "websockets",
        "xmlrpc.client",
    }
)
_PROCESS_MODULES = frozenset({"commands", "pty", "sh", "subprocess"})
_PROVIDER_MODULES = frozenset(
    {
        "anthropic",
        "azure.ai",
        "boto3",
        "botocore",
        "cohere",
        "google.cloud.aiplatform",
        "google.genai",
        "google.generativeai",
        "groq",
        "langchain",
        "litellm",
        "mistralai",
        "ollama",
        "openai",
        "replicate",
        "together",
        "vertexai",
    }
)
_AMBIENT_ENVIRONMENT_MODULES = frozenset(
    {
        "decouple",
        "dotenv",
        "environs",
        "keyring",
        "pydantic_settings",
    }
)
_AMBIENT_ENVIRONMENT_SYMBOLS = frozenset(
    {
        "dotenv_values",
        "environ",
        "environb",
        "get_keyring",
        "get_password",
        "getenv",
        "load_dotenv",
        "putenv",
        "unsetenv",
    }
)
_ALLOCATION_SYMBOLS = frozenset(
    {
        "_outcomes_for_allocation_rank",
        "allocate_balanced_outcomes",
        "allocation",
        "allocation_leaf",
        "allocation_seed",
        "allocations",
        "lineage_allocation",
        "lineageallocation",
        "validate_balanced_allocations",
    }
)
_ALLOCATION_CONSTANTS = frozenset(
    {
        "ALLOCATION_GOLDEN_DOMAIN",
        "ALLOCATION_ORDER_DOMAIN",
    }
)
_PROCESS_SYMBOLS = frozenset(
    {
        "Popen",
        "call",
        "check_call",
        "check_output",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "eval",
        "exec",
        "popen",
        "run",
        "system",
    }
)
_NETWORK_CALL_SYMBOLS = frozenset(
    {
        "create_connection",
        "create_datagram_endpoint",
        "open_connection",
        "start_server",
    }
)


def _matches_module(module: str, forbidden: frozenset[str]) -> bool:
    return any(module == name or module.startswith(f"{name}.") for name in forbidden)


def _module_category(module: str) -> str | None:
    if _matches_module(module, _NETWORK_MODULES):
        return "network"
    if _matches_module(module, _PROVIDER_MODULES):
        return "provider"
    root = module.split(".", maxsplit=1)[0].casefold()
    if root.startswith(("langchain_", "llama_index")):
        return "provider"
    if _matches_module(module, _PROCESS_MODULES):
        return "process"
    if _matches_module(module, _AMBIENT_ENVIRONMENT_MODULES):
        return "ambient-environment"
    if any(part.casefold() == "allocation" for part in module.split(".")):
        return "allocation"
    return None


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def _terminal_name(value: str) -> str:
    return value.rsplit(".", maxsplit=1)[-1]


def _is_allocation_symbol(value: str) -> bool:
    terminal = _terminal_name(value)
    folded = terminal.casefold()
    return (
        terminal in _ALLOCATION_CONSTANTS
        or folded in _ALLOCATION_SYMBOLS
        or folded.startswith("allocate_")
    )


def _is_process_symbol(value: str) -> bool:
    terminal = _terminal_name(value)
    if terminal in _PROCESS_SYMBOLS:
        return (
            (value == terminal and terminal in {"eval", "exec"})
            or value.startswith("os.")
            or value.startswith("posix.")
            or value.startswith("subprocess.")
            or value.startswith("asyncio.")
        )
    return value.startswith(
        ("os.exec", "os.spawn", "posix.exec", "posix.spawn")
    ) or value.startswith(("subprocess.", "commands.", "pty.", "sh."))


class _BoundaryVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._aliases: dict[str, str] = {}
        self._string_constants: dict[str, str] = {}
        self._findings: set[tuple[int, str, str]] = set()

    def findings(self) -> list[str]:
        return [
            f"{self._path}:{line}: {category}: {detail}"
            for line, category, detail in sorted(self._findings)
        ]

    def _record(self, node: ast.AST, category: str, detail: str) -> None:
        line = getattr(node, "lineno", 0)
        self._findings.add((line, category, detail))

    def _resolve(self, node: ast.AST) -> str | None:
        dotted = _dotted_name(node)
        if dotted is None:
            return None
        first, separator, suffix = dotted.partition(".")
        resolved = self._aliases.get(first, first)
        return f"{resolved}.{suffix}" if separator else resolved

    def _string_value(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self._string_constants.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._string_value(node.left)
            right = self._string_value(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    def _assignment_alias(self, node: ast.AST) -> str | None:
        resolved = self._resolve(node)
        if resolved is not None:
            return resolved
        if isinstance(node, ast.Call):
            function = self._resolve(node.func)
            if (
                function is not None
                and _terminal_name(function) == "getattr"
                and len(node.args) >= 2
            ):
                owner = self._resolve(node.args[0])
                member = self._string_value(node.args[1])
                if owner is not None and member is not None:
                    return f"{owner}.{member}"
        return None

    def _bind_assignment(self, target: ast.AST, value: ast.AST) -> None:
        if not isinstance(target, ast.Name):
            return
        resolved = self._assignment_alias(value)
        if resolved is None:
            self._aliases.pop(target.id, None)
        else:
            self._aliases[target.id] = resolved
        string_value = self._string_value(value)
        if string_value is None:
            self._string_constants.pop(target.id, None)
        else:
            self._string_constants[target.id] = string_value

    def _check_expression(self, node: ast.AST) -> None:
        resolved = self._resolve(node)
        if resolved is None:
            return
        if resolved.endswith(".SeedPurpose.ALLOCATION") or resolved == "SeedPurpose.ALLOCATION":
            self._record(node, "allocation-seed", "SeedPurpose.ALLOCATION is forbidden")
        if _is_allocation_symbol(resolved):
            self._record(node, "allocation", f"allocation symbol {_terminal_name(resolved)!r}")
        if _is_process_symbol(resolved):
            self._record(node, "process", f"external process access {_terminal_name(resolved)!r}")
        terminal = _terminal_name(resolved)
        if terminal in _NETWORK_CALL_SYMBOLS and resolved.startswith("asyncio."):
            self._record(node, "network", f"network access {terminal!r}")
        if terminal in _AMBIENT_ENVIRONMENT_SYMBOLS and (
            resolved.startswith("os.")
            or resolved.startswith("posix.")
            or resolved == terminal
            or any(resolved.startswith(f"{module}.") for module in _AMBIENT_ENVIRONMENT_MODULES)
        ):
            self._record(node, "ambient-environment", f"ambient access {terminal!r}")

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            local = imported.asname or imported.name.split(".", maxsplit=1)[0]
            self._aliases[local] = imported.name
            category = _module_category(imported.name)
            if category is not None:
                self._record(node, category, f"forbidden import {imported.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        category = _module_category(module)
        if category is not None:
            self._record(node, category, f"forbidden import {module!r}")
        for imported in node.names:
            if imported.name == "*":
                if module.endswith(".config") or module == "config":
                    self._record(node, "allocation", "wildcard config import is forbidden")
                continue
            full_name = f"{module}.{imported.name}" if module else imported.name
            local = imported.asname or imported.name
            self._aliases[local] = full_name
            imported_category = _module_category(full_name)
            if imported_category is not None and imported_category != category:
                self._record(
                    node,
                    imported_category,
                    f"forbidden import {full_name!r}",
                )
            if _is_allocation_symbol(imported.name):
                self._record(node, "allocation", f"allocation import {imported.name!r}")
            if _is_process_symbol(full_name):
                self._record(node, "process", f"process import {imported.name!r}")
            if imported.name in _AMBIENT_ENVIRONMENT_SYMBOLS and (
                module in {"os", "posix"} or _matches_module(module, _AMBIENT_ENVIRONMENT_MODULES)
            ):
                self._record(
                    node,
                    "ambient-environment",
                    f"ambient import {imported.name!r}",
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_assignment(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind_assignment(node.target, node.value)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            resolved = self._resolve(node)
            if resolved is not None and resolved.endswith("SeedPurpose"):
                self._record(
                    node,
                    "allocation-seed",
                    "bare SeedPurpose access is forbidden",
                )
            self._check_expression(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            owner = self._resolve(node.value)
            if owner is not None and owner.endswith("SeedPurpose"):
                if node.attr not in {"ID", "PUBLIC"}:
                    self._record(
                        node,
                        "allocation-seed",
                        "non-public SeedPurpose access is forbidden",
                    )
                self._check_expression(node)
                return
            self._check_expression(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        resolved = self._resolve(node.value)
        if resolved is not None and resolved.endswith("SeedPurpose"):
            member = self._string_value(node.slice)
            if member is None or member.casefold() == "allocation":
                self._record(
                    node,
                    "allocation-seed",
                    "dynamic SeedPurpose allocation lookup is forbidden",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        resolved = self._resolve(node.func)
        self._check_expression(node.func)
        terminal = _terminal_name(resolved) if resolved is not None else None

        if terminal == "SeedPurpose":
            purpose = self._string_value(node.args[0]) if node.args else None
            if purpose is None or purpose.casefold() == "allocation":
                self._record(
                    node,
                    "allocation-seed",
                    "SeedPurpose allocation construction is forbidden",
                )

        if terminal in {"__import__", "import_module"}:
            if node.args and (module := self._string_value(node.args[0])) is not None:
                category = _module_category(module)
                if category is not None:
                    self._record(node, category, f"dynamic forbidden import {module!r}")
            else:
                self._record(node, "dynamic-import", "non-literal dynamic import is forbidden")

        if terminal == "getattr" and len(node.args) >= 2:
            owner = self._resolve(node.args[0])
            member = self._string_value(node.args[1])
            if member is not None:
                if (
                    owner is not None
                    and owner.endswith("SeedPurpose")
                    and member.casefold() == "allocation"
                ):
                    self._record(
                        node,
                        "allocation-seed",
                        "dynamic SeedPurpose.ALLOCATION access is forbidden",
                    )
                if _is_allocation_symbol(member):
                    self._record(
                        node,
                        "allocation",
                        f"dynamic allocation access {member!r}",
                    )
                if owner in {"os", "posix"} and member in _AMBIENT_ENVIRONMENT_SYMBOLS:
                    self._record(
                        node,
                        "ambient-environment",
                        f"dynamic ambient access {member!r}",
                    )
            elif owner is not None:
                if owner.endswith("SeedPurpose"):
                    self._record(node, "allocation-seed", "dynamic SeedPurpose access is forbidden")
                elif owner.endswith(".config") or owner == "config":
                    self._record(node, "allocation", "dynamic config access is forbidden")
                elif owner in {"os", "posix"}:
                    self._record(node, "ambient-environment", "dynamic ambient access is forbidden")
                elif owner in {"importlib", "builtins"}:
                    self._record(node, "dynamic-import", "dynamic import access is forbidden")
        self.generic_visit(node)


def scan_source(path: Path, source: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=str(path), mode="exec")
    except (SyntaxError, ValueError) as error:
        line = error.lineno if isinstance(error, SyntaxError) and error.lineno is not None else 0
        return [f"{path}:{line}: parse: invalid Python source"]
    visitor = _BoundaryVisitor(path)
    visitor.visit(tree)
    return visitor.findings()


def _boundary_paths(root: Path) -> tuple[Path, ...]:
    package = root / PACKAGE_DIRECTORY
    declared = {package / name for name in BOUNDARY_MODULES}
    if package.is_dir():
        declared.update(package.glob("review*.py"))
    return tuple(sorted(path for path in declared if path.is_file() or path.is_symlink()))


def validate_public_review_boundary(root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    for path in _boundary_paths(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append(f"{relative}:0: read: symbolic-link source is forbidden")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(f"{relative}:0: read: Python source is unreadable")
            continue
        findings.extend(scan_source(relative, source))
    return sorted(findings)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) > 1:
        print("usage: check_public_review_boundary.py [ROOT]", file=sys.stderr)
        return 2
    root = Path(arguments[0]).resolve() if arguments else ROOT
    findings = validate_public_review_boundary(root)
    if findings:
        for finding in findings:
            print(finding, file=sys.stderr)
        return 1
    print("Public review boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
