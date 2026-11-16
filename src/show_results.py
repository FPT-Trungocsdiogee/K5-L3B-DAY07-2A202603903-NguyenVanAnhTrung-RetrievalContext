"""Print saved retrieval results (summary.json + traces.json) on the terminal.

Examples (from the repo root):
    python src/show_results.py starter/results/dev-*                 # compare all dev runs
    python src/show_results.py starter/results/dev-dense --details   # per-question table
    python src/show_results.py starter/results/dev-dense --misses    # retrieved text for each miss

Glob patterns are expanded by the script, so they also work in PowerShell.
Works for runs from src/run_retrieval.py and from starter/baseline.py.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import print_comparison, print_run  # noqa: E402


def expand(patterns: list[str]) -> list[Path]:
    folders: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        folders += [Path(m) for m in matches if (Path(m) / 'summary.json').is_file()]
    return list(dict.fromkeys(folders))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', help='result folders or glob patterns')
    p.add_argument('--details', action='store_true', help='per-question table for each run')
    p.add_argument('--misses', action='store_true', help='show gold evidence and retrieved text for misses')
    p.add_argument('--queries', action='store_true', help='show transformed queries')
    p.add_argument('--split', choices=('dev', 'test'), help='only show runs of this split')
    args = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    folders = expand(args.runs)
    if not folders:
        sys.exit(f'no folder with summary.json matches: {" ".join(args.runs)}')
    runs = []
    for folder in folders:
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        if args.split and summary.get('split') != args.split:
            continue
        runs.append((folder, summary))
    if not runs:
        sys.exit(f'no run with split={args.split}')

    if args.details or args.misses or args.queries:
        for folder, summary in runs:
            traces = json.loads((folder / 'traces.json').read_text(encoding='utf-8'))
            absent_file = folder / 'absent_traces.json'
            absent = json.loads(absent_file.read_text(encoding='utf-8')) if absent_file.is_file() else None
            print(f'\n===== {folder} =====')
            print_run(summary, traces, details=args.details or args.queries, misses=args.misses,
                      show_queries=args.queries, absent_traces=absent)
    if len(runs) > 1 or not (args.details or args.misses or args.queries):
        print()
        print_comparison([(f.name, s) for f, s in runs])


if __name__ == '__main__':
    main()
