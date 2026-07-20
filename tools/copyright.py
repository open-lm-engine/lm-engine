# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import os
import re
import subprocess
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--repo", type=str, required=True)
parser.add_argument("--exclude", type=str, required=False)
parser.add_argument("--header", type=str, required=True)
parser.add_argument("--extra-name", type=str, required=False)
parser.add_argument("--no-contributors", action="store_true", required=False)
parser.add_argument("--check", action="store_true", required=False)
args = parser.parse_args()


_CPP_LIKE_EXTENSIONS = [".cu", ".h", ".c", ".cpp"]
_PYTHON_LIKE_EXTENSIONS = [
    ".py",
    ".yml",
    ".yaml",
    ".clang-format",
    "requirements-dev.txt",
    "requirements.txt",
    "setup.cfg",
    "Makefile",
]
_HTML_LIKE_EXTENSIONS = [".html", ".md"]

_BANNED = [".git"]
if args.exclude:
    exclude = open(args.exclude, "r").readlines()
    exclude = [i.strip() for i in exclude]
    _BANNED.extend(exclude)

_BANNED = [os.path.realpath(i) for i in _BANNED]


def _make_header(header: str, comment_char: str) -> str:
    header = header.split("\n")
    if comment_char:
        header = [f"{comment_char} {i}" for i in header]
    header = "\n".join(header)
    return header + "\n"


def _build_author_map(repo: str) -> dict[str, dict[str, int]]:
    """Run git log once and return {abs_path: {author: commit_count}}."""
    try:
        result = subprocess.run(
            ["git", "-C", repo, "log", "--name-only", "--format=%x00%an"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {}
    author_map: dict[str, dict[str, int]] = {}
    current_author = None
    for line in result.stdout.splitlines():
        if line.startswith("\x00"):
            current_author = line[1:]
        elif line and current_author:
            abs_path = os.path.realpath(os.path.join(repo, line))
            counts = author_map.setdefault(abs_path, {})
            counts[current_author] = counts.get(current_author, 0) + 1
    return author_map


_AUTHOR_MAP: dict[str, dict[str, int]] = {}


def _get_git_authors(file: str) -> list[str]:
    counts = _AUTHOR_MAP.get(os.path.realpath(file), {})
    return sorted(counts.keys())


def _resolve_copyright_line(file: str) -> str:
    if args.no_contributors:
        if args.extra_name:
            return args.header.replace("__authors__", args.extra_name)
        return args.header.replace(", __authors__", "").replace("__authors__", "")
    authors = _get_git_authors(file)
    if args.extra_name:
        authors = [args.extra_name] + sorted(a for a in authors if a != args.extra_name)
    if authors:
        return args.header.replace("__authors__", ", ".join(authors))
    return args.header.replace(", __authors__", "").replace("__authors__", "")


# Structural patterns — flexible on year and author content
_CPP_PATTERN = re.compile(r"// \*+\n// Copyright[^\n]*\n// \*+\n\n")
_PYTHON_PATTERN = re.compile(r"# \*+\n# Copyright[^\n]*\n# \*+\n\n")
_HTML_PATTERN = re.compile(r"<!-- \*+\n\s*Copyright[^\n]*\n\*+ -->\n\n")


def _build_cpp_header(file: str) -> str:
    return (
        "// **************************************************\n"
        f"{_make_header(_resolve_copyright_line(file), '//')}"
        "// **************************************************\n\n"
    )


def _build_python_header(file: str) -> str:
    return (
        "# **************************************************\n"
        f"{_make_header(_resolve_copyright_line(file), '#')}"
        "# **************************************************\n\n"
    )


def _build_html_header(file: str) -> str:
    return (
        "<!-- **************************************************\n"
        f"{_make_header(_resolve_copyright_line(file), '')}"
        "************************************************** -->\n\n"
    )


def _check_and_add_copyright_header(file: str, build_header_fn, pattern: re.Pattern) -> bool:
    code = open(file, "r").read()

    if len(code) == 0:
        return True

    if args.check:
        return bool(pattern.match(code))

    header = build_header_fn(file)
    code_stripped = pattern.sub("", code)
    if code_stripped != code:
        code = f"{header}{code_stripped}"
    elif not code.startswith(header):
        code = f"{header}{code}"

    open(file, "w").writelines([code])

    return True


def _is_banned(path: str) -> bool:
    assert not path.endswith("/")

    for banned_directory in _BANNED:
        if path == banned_directory or path.startswith(banned_directory + os.sep):
            return True

    return False


directory = os.path.realpath(args.repo)
_AUTHOR_MAP = {} if args.no_contributors else _build_author_map(directory)

missing = []
for root, dirs, files in os.walk(directory):
    if _is_banned(root):
        continue

    for file in files:
        file = os.path.join(root, file)

        if _is_banned(file):
            continue

        ok = True
        if any([file.endswith(i) for i in _CPP_LIKE_EXTENSIONS]):
            ok = _check_and_add_copyright_header(file, _build_cpp_header, _CPP_PATTERN)
        elif any([file.endswith(i) for i in _PYTHON_LIKE_EXTENSIONS]):
            ok = _check_and_add_copyright_header(file, _build_python_header, _PYTHON_PATTERN)
        elif any([file.endswith(i) for i in _HTML_LIKE_EXTENSIONS]):
            ok = _check_and_add_copyright_header(file, _build_html_header, _HTML_PATTERN)

        if not ok:
            missing.append(os.path.relpath(file, directory))

if missing:
    for f in sorted(missing):
        print(f"No copyright found on '{f}'.")
    raise SystemExit(1)
