#!/usr/bin/env python3
"""Pre-commit hook to check that all imports are listed in requirements.txt.

This script validates that all third-party imports in Python files
have corresponding entries in requirements.txt.
"""

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Standard library modules (partial list - extend as needed)
STDLIB_MODULES = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "builtins",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "functools",
    "glob",
    "gzip",
    "hashlib",
    "heapq",
    "html",
    "http",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "pprint",
    "queue",
    "random",
    "re",
    "setuptools",
    "shutil",
    "signal",
    "socket",
    "sqlite3",
    "statistics",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "traceback",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "xml",
    "zipfile",
    "zlib",
}

# Internal project modules (adjust based on your package structure)
INTERNAL_MODULES = {"data", "model", "train", "inference", "utils", "config"}


def extract_imports_from_file(filepath: Path) -> Set[str]:
    """Extract top-level module names from import statements.

    Args:
        filepath: Path to Python file

    Returns:
        Set of top-level module names
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                imports.add(top_level)
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (from .module or from ..module)
            if node.level > 0:
                continue
            if node.module:
                top_level = node.module.split(".")[0]
                imports.add(top_level)

    return imports


def extract_packages_from_requirements(requirements_file: Path) -> Set[str]:
    """Extract package names from requirements.txt.

    Args:
        requirements_file: Path to requirements.txt

    Returns:
        Set of package names
    """
    if not requirements_file.exists():
        return set()

    packages = set()
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Extract package name (before version specifiers)
            package = line.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
            # Handle package name normalization (e.g., 'opencv-python' -> 'cv2')
            package_lower = package.lower()
            packages.add(package_lower)

            # Add common package name variations
            if package_lower == "opencv-python":
                packages.add("cv2")
            elif package_lower == "pillow":
                packages.add("pil")
            elif package_lower == "pyyaml":
                packages.add("yaml")
            elif package_lower == "python-dotenv":
                packages.add("dotenv")
            elif package_lower == "scikit-image":
                packages.add("skimage")
            elif package_lower == "scikit-learn":
                packages.add("sklearn")

    return packages


def check_imports(files: List[str]) -> Tuple[bool, List[str]]:
    """Check if all imports are in requirements.txt.

    Args:
        files: List of Python file paths to check

    Returns:
        Tuple of (success, list of error messages)
    """
    repo_root = Path(__file__).parent.parent
    requirements_file = repo_root / "requirements.txt"
    requirements_dev_file = repo_root / "requirements-dev.txt"

    # Get all allowed packages
    required_packages = extract_packages_from_requirements(requirements_file)
    dev_packages = extract_packages_from_requirements(requirements_dev_file)
    all_allowed = required_packages | dev_packages | STDLIB_MODULES | INTERNAL_MODULES

    errors = []
    for filepath in files:
        path = Path(filepath)
        if not path.exists() or path.suffix != ".py":
            continue

        imports = extract_imports_from_file(path)

        # Check each import
        for imp in imports:
            imp_lower = imp.lower()
            if imp_lower not in all_allowed and imp not in STDLIB_MODULES:
                errors.append(
                    f"{filepath}: Import '{imp}' not found in "
                    f"requirements.txt or requirements-dev.txt"
                )

    return len(errors) == 0, errors


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: check_imports.py <file1> <file2> ...")
        sys.exit(0)

    files = sys.argv[1:]
    success, errors = check_imports(files)

    if not success:
        print("❌ Import check failed!")
        print("\nThe following imports are not in requirements.txt:\n")
        for error in errors:
            print(f"  {error}")
        print("\nPlease add missing packages to requirements.txt or requirements-dev.txt")
        sys.exit(1)

    print("✅ All imports are properly declared in requirements.txt")
    sys.exit(0)


if __name__ == "__main__":
    main()
