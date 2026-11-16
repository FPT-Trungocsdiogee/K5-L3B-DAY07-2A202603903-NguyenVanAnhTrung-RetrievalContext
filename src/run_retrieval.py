"""Evaluate dense / lexical / hybrid retrieval, optionally with query transformation.

Output (traces.json, summary.json) uses the same metrics as starter/baseline.py:
Hit@1, Evidence Hit@3 (hit within top-k) and Context Evidence Rate.

Examples (from the repo root):
    python src/run_retrieval.py --method lexical --split dev                  # -> starter/results/dev-lexical
    python src/run_retrieval.py --method hybrid  --transform multi --split dev --use-filter
                                                                              # -> starter/results/dev-hybrid-multi-filter
    python src/run_retrieval.py --method dense   --split dev --output my_runs/dense-k5 --top-k 5   # your own folder
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
sys.path.insert(0, str(ROOT / 'src'))
from baseline import hit, pack_context, validate  # noqa: E402
from query_transform import MODES, QueryTransformer, TransformedRetriever, openrouter_llm  # noqa: E402
from report import print_run  # noqa: E402
from retrieval import DenseRetriever, HybridRetriever, LexicalRetriever  # noqa: E402

DEFAULT_DATA = ROOT / 'starter' / 'data' / 'converted'


def default_output(args) -> Path:
    parts = [args.split, args.method]
    if args.transform != 'none':
        parts.append(args.transform)
    if args.use_filter:
        parts.append('filter')
    return ROOT / 'starter' / 'results' / '-'.join(parts)


def load_env() -> None:
    from dotenv import load_dotenv  # earlier files win: load_dotenv never overrides set values
    for env in (ROOT / '.env', ROOT / 'starter' / '.env'):
        if env.is_file():
            load_dotenv(env)


def build(args, chunks):
    lexical = LexicalRetriever(chunks, k1=args.k1, b=args.b)
    if args.method == 'lexical':
        retriever = lexical
    else:
        dense = DenseRetriever(args.strategy)
        retriever = dense if args.method == 'dense' else HybridRetriever(
            dense, lexical, pool=args.pool, k=args.rrf_k, weights=(args.dense_weight, args.lexical_weight))
    if args.transform == 'none':
        return retriever, None
    cache = args.output / 'transform_cache.json'
    transformer = QueryTransformer(args.transform, llm=openrouter_llm(args.transform_model), n=args.n_queries,
                                   cache_path=cache)
    return TransformedRetriever(retriever, transformer, pool=args.pool, k=args.rrf_k), transformer


def evaluate(retriever, questions, args):
    traces = []
    label = args.method + ('' if args.transform == 'none' else f'+{args.transform}')
    for q in questions:
        filters = q.get('filter') if args.use_filter else None
        start = time.perf_counter()
        results = retriever.search(q['question'], filters=filters, top_k=args.top_k)
        context, ids = pack_context(results, args.budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'],
                       'queries': getattr(retriever, 'last_queries', [q['question']]),
                       'strategy': label, 'filter': filters, 'top_k': args.top_k, 'budget_chars': args.budget,
                       'gold': q['gold'], 'results': marked, 'context': context, 'context_ids': ids,
                       'context_chars': len(context), 'elapsed_ms': elapsed_ms,
                       'hit1': bool(marked and marked[0]['hit']),
                       'hit3': any(c['hit'] for c in marked),
                       'context_hit': any(c['hit'] and c['id'] in ids for c in marked)})
    return traces, label


def load_agent_cases(path: Path, split: str) -> list[dict]:
    """Questions without evidence (answerable=false) from benchmark.json `agent_cases`."""
    if not path.is_file():
        print(f'[warn] {path} not found; agent_cases skipped')
        return []
    data = json.loads(path.read_text(encoding='utf-8'))
    cases = data.get('agent_cases', []) if isinstance(data, dict) else []
    selected, ids = [], set()
    for c in cases:
        if c.get('split') != split:
            continue  # never read the other split
        if not isinstance(c.get('id'), str) or c['id'] in ids or not isinstance(c.get('question'), str) \
                or not c['question'].strip():
            raise ValueError(f'agent_case {c.get("id")!r}: id must be unique and question non-empty')
        ids.add(c['id'])
        if c.get('answerable') is not False:
            print(f"[warn] agent_case {c['id']}: answerable is not false; skipped (no gold to score it)")
            continue
        selected.append(c)
    return selected


def evaluate_absent(retriever, cases, args, label):
    """Run unanswerable questions through the same retriever. There is no gold, so no hit metric;
    the trace records what reaches the context (absence_reason is never sent to retrieval)."""
    traces = []
    for c in cases:
        filters = c.get('filter') if args.use_filter else None
        start = time.perf_counter()
        results = retriever.search(c['question'], filters=filters, top_k=args.top_k)
        context, ids = pack_context(results, args.budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        traces.append({'query_id': c['id'], 'split': c['split'], 'question': c['question'], 'answerable': False,
                       'expected_behavior': c.get('expected_behavior'), 'absence_reason': c.get('absence_reason'),
                       'queries': getattr(retriever, 'last_queries', [c['question']]),
                       'strategy': label, 'filter': filters, 'top_k': args.top_k, 'budget_chars': args.budget,
                       'results': results, 'context': context, 'context_ids': ids,
                       'context_chars': len(context), 'elapsed_ms': elapsed_ms,
                       'top1_score': results[0]['score'] if results else None})
    return traces


def absent_summary(answerable_traces, absent_traces):
    """Compare top-1 scores: absent cases scoring below every answerable question could be
    rejected by a score threshold. Scores are only comparable within one method/config."""
    ans = [t['results'][0]['score'] for t in answerable_traces if t['results']]
    absent = [t['top1_score'] for t in absent_traces if t['top1_score'] is not None]
    floor = min(ans) if ans else None
    return {'n': len(absent_traces),
            'empty_results': sum(t['top1_score'] is None for t in absent_traces),
            'top1_score_median': statistics.median(absent) if absent else None,
            'top1_score_max': max(absent) if absent else None,
            'answerable_top1_score_median': statistics.median(ans) if ans else None,
            'answerable_top1_score_min': floor,
            'below_answerable_min': (sum(s < floor for s in absent) if floor is not None else None),
            'note': 'retrieval cannot judge abstention; label abstention_correct on the agent run'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True, choices=('dense', 'lexical', 'hybrid'))
    p.add_argument('--transform', default='none', choices=MODES)
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--output', type=Path, default=None,
                   help='output folder (any path you choose); default: starter/results/<split>-<method>'
                        '[-<transform>][-filter]. Same-name files are overwritten')
    p.add_argument('--chunks', type=Path, default=DEFAULT_DATA / 'chunks.json')
    p.add_argument('--questions', type=Path, default=DEFAULT_DATA / 'questions.json')
    p.add_argument('--cases', type=Path, default=ROOT / 'benchmark' / 'benchmark.json',
                   help='file with `agent_cases` (answerable=false questions)')
    p.add_argument('--no-agent-cases', action='store_true', help='skip agent_cases')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--budget', type=int, default=800, help='context budget in characters')
    p.add_argument('--use-filter', action='store_true', help="apply each question's metadata filter")
    p.add_argument('--strategy', default='B', help='Day 6 index for dense retrieval')
    p.add_argument('--pool', type=int, default=20, help='candidates per list before RRF fusion')
    p.add_argument('--rrf-k', type=int, default=60)
    p.add_argument('--dense-weight', type=float, default=1.0)
    p.add_argument('--lexical-weight', type=float, default=1.0)
    p.add_argument('--k1', type=float, default=1.5, help='BM25 k1')
    p.add_argument('--b', type=float, default=0.75, help='BM25 b')
    p.add_argument('--n-queries', type=int, default=3, help='paraphrases for --transform multi')
    p.add_argument('--transform-model', default=None, help='default: TRANSFORM_MODEL, CHAT_MODEL or openai/gpt-4o-mini')
    out = p.add_argument_group('terminal output')
    out.add_argument('--quiet', action='store_true', help='print only the summary, no per-question table')
    out.add_argument('--misses', action='store_true', help='show gold evidence and retrieved text for misses')
    out.add_argument('--show-queries', action='store_true', help='show transformed queries per question')
    out.add_argument('--json', action='store_true', help='print summary.json as raw JSON instead of tables')
    args = p.parse_args()
    if args.top_k < 1 or args.budget < 1 or args.pool < 1:
        p.error('--top-k, --budget and --pool must be >= 1')
    if args.output is None:
        args.output = default_output(args)

    sys.stdout.reconfigure(encoding='utf-8')
    load_env()
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == args.split]  # never read the other split
    validate(chunks, selected)
    cases = [] if args.no_agent_cases else load_agent_cases(args.cases, args.split)

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    retriever, transformer = build(args, chunks)
    index_seconds = time.perf_counter() - started
    traces, label = evaluate(retriever, selected, args)
    absent_traces = evaluate_absent(retriever, cases, args, label)

    n = len(traces)
    counts = {key: sum(t[key] for t in traces) for key in ('hit1', 'hit3', 'context_hit')}
    summary = {'strategy': label, 'method': args.method, 'transform': args.transform, 'split': args.split, 'n': n,
               **counts, **{f'{k}_rate': round(v / n, 4) for k, v in counts.items()},
               'config': {'top_k': args.top_k, 'budget_chars': args.budget, 'use_filter': args.use_filter,
                          'day6_index': args.strategy if args.method != 'lexical' else None,
                          'pool': args.pool, 'rrf_k': args.rrf_k,
                          'weights': {'dense': args.dense_weight, 'lexical': args.lexical_weight},
                          'bm25': {'k1': args.k1, 'b': args.b},
                          'transform_model': getattr(getattr(transformer, 'llm', None), 'model', None),
                          'n_queries': args.n_queries if args.transform == 'multi' else None},
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'index_seconds': index_seconds,
               'service_cost_usd': round(getattr(retriever, 'cost_usd', 0.0), 6),
               'transform_usage': transformer.usage if transformer else None,
               'absent': absent_summary(traces, absent_traces) if absent_traces else None}
    outputs = [('traces.json', traces), ('summary.json', summary)]
    if absent_traces:
        outputs.append(('absent_traces.json', absent_traces))
    else:
        (args.output / 'absent_traces.json').unlink(missing_ok=True)  # do not leave a stale file
    for name, obj in outputs:
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_run(summary, traces, details=not args.quiet, misses=args.misses, show_queries=args.show_queries,
                  absent_traces=absent_traces)
    print(f"\nSaved {', '.join(name for name, _ in outputs)} to {args.output}")


if __name__ == '__main__':
    main()
