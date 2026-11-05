"""NIST 4 mm square-peg insertion task, registered as ``PegInset``."""

from . import tasks  # noqa: F401


def main() -> None:
    """Run Isaac Lab commands with PegInset registered."""
    from isaaclab.__main__ import main as isaaclab_main

    isaaclab_main()
