"""`dmt` command line.

dmt version                 Print the toolkit version
dmt pins                    Print the tool versions the toolkit pins
dmt check-sync REPO [...]   Compare each repo's uv.lock with those pins;
                            exit 1 if any repo drifts or lacks the toolkit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_ml_toolkit import __version__
from data_ml_toolkit.sync import check_repo, pinned_tools


def _check_sync(repos: list[Path]) -> int:
    pins = pinned_tools()
    tools = sorted(pins)
    header = ["repo", "toolkit", *tools]
    rows = [["(pinned)", __version__, *(pins[t] for t in tools)]]
    status = 0
    problems: list[str] = []
    for repo in repos:
        report = check_repo(repo, pins)
        rows.append(
            [
                report.repo,
                report.toolkit or "missing",
                *(report.locked[t] or "-" for t in tools),
            ]
        )
        if report.toolkit is None:
            problems.append(f"{report.repo}: does not depend on data-ml-toolkit")
        for tool, (locked, pinned) in report.drift.items():
            problems.append(f"{report.repo}: {tool} {locked} (toolkit pins {pinned})")
        if not report.in_sync:
            status = 1

    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())
    print()
    print("\n".join(problems) if problems else "All repos in sync.")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dmt", description="data-ml-toolkit utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the toolkit version")
    sub.add_parser("pins", help="print the pinned tool versions")
    check = sub.add_parser("check-sync", help="compare repos' uv.lock with the pins")
    check.add_argument("repos", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "pins":
        for tool, version in sorted(pinned_tools().items()):
            print(f"{tool}=={version}")
        return 0
    return _check_sync(args.repos)


if __name__ == "__main__":
    sys.exit(main())
