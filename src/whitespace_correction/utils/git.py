import subprocess
from pathlib import Path


def git_branch() -> str:
    return subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=Path(__file__).resolve().parent.parent.parent
    ).strip().decode("utf8")


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent.parent
    ).strip().decode("utf8")
