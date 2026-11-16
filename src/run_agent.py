"""Run the real agent (starter/agentic.py) on a benchmark split, save its trace and print the report tables.

For every question of the split (answerable `questions` + unanswerable `agent_cases`) the script runs
`run_agent` with the real OpenRouter `Controller` and a retriever, then prints a step-by-step trace
(each search query, the chunks it returned, what reached the context, the final answer and citations).

Saved in the output folder (the formats starter/agent_metrics.py expects):
    cases.json             id, split, answerable
    traces.json            query_id, status (answered/abstain/error), events, citations + trace details
    summary.json           config, counts, model usage and cost
    reviews_template.json  empty human labels for this run (always rewritten)
    reviews.json           created from the template only if it does not exist yet; fill it in by hand

Then the two "Agent và đánh giá" tables of the report are printed as Markdown. Label columns
(grounded answer, citation validity, abstention) stay "chưa chấm" until reviews.json is fully filled in.

Examples (from the repo root):
    python src/run_agent.py --split dev                                  # -> starter/results/agent-dev
    python src/run_agent.py --split dev --method hybrid --use-filter     # -> starter/results/agent-dev-hybrid-filter
    python src/run_agent.py --split dev --only q01,a-q01                 # a few questions, for debugging
    python src/run_agent.py --show starter/results/agent-dev             # reprint a saved run (after labelling)
    python starter/agent_metrics.py --cases starter/results/agent-dev/cases.json \\
        --traces starter/results/agent-dev/traces.json --reviews starter/results/agent-dev/reviews.json \\
        --split dev --output starter/results/agent-dev/metrics.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
sys.path.insert(0, str(ROOT / 'src'))
from baseline import hit  # noqa: E402
from report import short  # noqa: E402

DEFAULT_DATA = ROOT / 'starter' / 'data' / 'converted'
PENDING = 'chưa chấm'


# ---------------------------------------------------------------- recording wrappers

class TracingController:
    """Wraps agentic.Controller and keeps every raw decision (run_agent drops rejected answers)."""

    def __init__(self, controller) -> None:
        self.inner = controller
        self.model = controller.model
        self.usage = controller.usage
        self.steps: list[dict[str, Any]] = []

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        before = dict(self.inner.usage)
        start = time.perf_counter()
        action = self.inner.decide(question, history)
        self.steps.append({'kind': 'decide', 'action': action, 'ms': (time.perf_counter() - start) * 1000,
                           'prompt_tokens': self.inner.usage['prompt_tokens'] - before['prompt_tokens'],
                           'completion_tokens': self.inner.usage['completion_tokens'] - before['completion_tokens'],
                           'cost_usd': self.inner.usage['cost_usd'] - before['cost_usd']})
        return action


class TracingRetriever:
    """Wraps any retriever with search(query, *, filters, top_k) and times each call."""

    def __init__(self, retriever, controller: TracingController) -> None:
        self.inner = retriever
        self.controller = controller  # shares the step list so decisions and searches stay in order

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        start = time.perf_counter()
        results = self.inner.search(query, filters=filters, top_k=top_k)
        self.controller.steps.append({'kind': 'search', 'query': query, 'ms': (time.perf_counter() - start) * 1000,
                                      'n_results': len(results)})
        return results


# ---------------------------------------------------------------- data

def load_env() -> None:
    from dotenv import load_dotenv  # earlier files win: load_dotenv never overrides set values
    for env in (ROOT / '.env', ROOT / 'starter' / '.env'):
        if env.is_file():
            load_dotenv(env)


def load_cases(questions_path: Path, benchmark_path: Path, split: str) -> list[dict[str, Any]]:
    """Answerable questions (converted, with gold) + unanswerable agent_cases of one split only."""
    benchmark = json.loads(benchmark_path.read_text(encoding='utf-8'))
    types = {q['id']: q.get('type') for q in benchmark.get('questions', [])}
    cases = []
    for q in json.loads(questions_path.read_text(encoding='utf-8')):
        if q.get('split') == split:
            cases.append({**q, 'type': q.get('type') or types.get(q['id']), 'answerable': True})
    for c in benchmark.get('agent_cases', []):
        if c.get('split') == split:
            if c.get('answerable') is not False:
                raise ValueError(f"agent_case {c.get('id')}: answerable must be false")
            cases.append({**c, 'gold': []})
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate case id across questions and agent_cases')
    return cases


def build_retriever(args):
    from retrieval import DenseRetriever, HybridRetriever, LexicalRetriever
    if args.method == 'dense':
        return DenseRetriever(args.strategy)
    lexical = LexicalRetriever(json.loads(args.chunks.read_text(encoding='utf-8')))
    if args.method == 'lexical':
        return lexical
    return HybridRetriever(DenseRetriever(args.strategy), lexical, pool=args.pool, k=args.rrf_k)


def default_output(args) -> Path:
    parts = ['agent', args.split] + ([] if args.method == 'dense' else [args.method])
    if args.use_filter:
        parts.append('filter')
    return ROOT / 'starter' / 'results' / '-'.join(parts)


# ---------------------------------------------------------------- run

def run_case(case, controller, retriever, args) -> dict[str, Any]:
    from agentic import run_agent
    tracer = TracingController(controller)
    filters = case.get('filter') if args.use_filter else None
    start = time.perf_counter()
    base = {'query_id': case['id'], 'split': case['split'], 'type': case.get('type'),
            'answerable': case['answerable'], 'question': case['question'], 'filter': filters}
    try:
        out = run_agent(case['question'], tracer, TracingRetriever(retriever, tracer), filters=filters,
                        max_searches=args.max_searches, top_k=args.top_k, budget=args.budget)
    except Exception as error:  # provider/DB failure: recorded, excluded from quality metrics
        return {**base, 'status': 'error', 'error': f'{type(error).__name__}: {error}', 'answer': None,
                'citations': [], 'events': [], 'steps': tracer.steps,
                'elapsed_ms': (time.perf_counter() - start) * 1000}
    gold = case.get('gold') or []
    for e in out['events']:
        if e.get('event') == 'search':
            e['results'] = [{**r, 'hit': hit(r, gold)} for r in e['results']]
    searches = [e for e in out['events'] if e.get('event') == 'search']
    final_ids = searches[-1]['context_ids'] if searches else []
    gold_ids = {r['id'] for e in searches for r in e['results'] if r['hit']}
    last = next((s['action'] for s in reversed(tracer.steps) if s['kind'] == 'decide'), {})
    return {**base, 'status': out['status'], 'answer': out['answer'], 'citations': out['citations'],
            'events': out['events'], 'steps': tracer.steps,
            'model_answer': last.get('answer') if last.get('action') == 'answer' else None,
            'model_citations': last.get('citations') if last.get('action') == 'answer' else None,
            'n_searches': len(searches), 'n_model_calls': sum(s['kind'] == 'decide' for s in tracer.steps),
            'expected_answer': case.get('expected_answer'),
            # automatic signals only; they never replace the human labels in reviews.json
            'auto': {'evidence_retrieved': bool(gold_ids), 'evidence_in_final_context': bool(gold_ids & set(final_ids)),
                     'cites_evidence_chunk': bool(gold_ids & set(out['citations']))} if case['answerable'] else None,
            'elapsed_ms': (time.perf_counter() - start) * 1000}


def review_template(traces) -> list[dict[str, Any]]:
    rows = []
    for t in traces:
        if t['status'] == 'error':
            continue
        row = {'query_id': t['query_id'], 'citation_support': [None] * len(t['citations']), 'reason': '',
               '_status': t['status'], '_question': t['question'], '_answer': t['answer'],
               '_citations': t['citations']}
        if t['answerable']:
            row['answer_grade'] = 0 if t['status'] == 'abstain' else None  # abstain on answerable must be 0
            row['_expected_answer'] = t.get('expected_answer')
        else:
            row['abstention_correct'] = None
        rows.append(row)
    return rows


# ---------------------------------------------------------------- terminal trace

def print_trace(t: dict[str, Any], width: int) -> None:
    kind = 'answerable' if t['answerable'] else 'no evidence'
    flt = f" filter={json.dumps(t['filter'], ensure_ascii=False)}" if t.get('filter') else ''
    print(f"\n=== {t['query_id']}  [{t['split']} | {t.get('type') or '-'} | {kind}]{flt}")
    print(f"Q: {short(t['question'], width)}")
    searches = iter(e for e in t['events'] if e.get('event') == 'search')
    step = 0
    for s in t['steps']:
        if s['kind'] != 'decide':
            continue
        step += 1
        a = s['action']
        cost = f"{s['ms'] / 1000:.1f}s, {s['prompt_tokens']}+{s['completion_tokens']} tok"
        if a.get('action') == 'search':
            e = next(searches, None)
            print(f"  [{step}] search  \"{short(a.get('query') or '', width - 20)}\"   ({cost})")
            if e is None:  # rejected by the search limit
                print('        (not executed: search limit reached)')
                continue
            for rank, r in enumerate(e['results'], start=1):
                ctx = 'ctx' if r['id'] in e['context_ids'] else '   '
                flag = '  EVIDENCE' if r.get('hit') else ''
                print(f"        #{rank} {ctx} p{r['page']} score={r['score']:.4f} {r['id']}{flag}")
                print(f"              {short(r['text'], width - 14)}")
            print(f"        context: {len(e['context'])} chars, {len(e['context_ids'])} chunk(s)")
        else:
            print(f"  [{step}] answer  ({cost})")
            print(f"        model: {short(a.get('answer') or '', width - 15)}")
            print(f"        model citations: {a.get('citations')}")
    for e in t['events']:
        if e.get('event') not in ('search', 'answer'):
            print(f"  -> {e['event']}")
    if t['status'] == 'error':
        print(f"  ERROR: {t['error']}")
    else:
        print(f"  => {t['status'].upper()}: {short(t['answer'] or '', width - 15)}")
        print(f"     citations: {t['citations'] or '[]'}")
    if t.get('auto'):
        a = t['auto']
        yn = lambda v: 'yes' if v else 'no'  # noqa: E731
        print(f"     auto: evidence retrieved={yn(a['evidence_retrieved'])}, in final context="
              f"{yn(a['evidence_in_final_context'])}, cites evidence chunk={yn(a['cites_evidence_chunk'])}")
    print(f"     {t.get('n_searches', 0)} search, {t.get('n_model_calls', 0)} model call(s), "
          f"{t['elapsed_ms'] / 1000:.1f}s")


# ---------------------------------------------------------------- report tables

def labels_complete(traces, reviews) -> tuple[bool, list[str]]:
    by_id = {r.get('query_id'): r for r in reviews or []}
    missing = []
    for t in traces:
        if t['status'] == 'error':
            continue
        r = by_id.get(t['query_id'])
        ok = (r is not None and isinstance(r.get('citation_support'), list)
              and len(r['citation_support']) == len(t['citations'])
              and all(type(v) is bool for v in r['citation_support'])
              and (type(r.get('answer_grade')) is int if t['answerable'] else type(r.get('abstention_correct')) is bool))
        if not ok:
            missing.append(t['query_id'])
    return not missing, missing


def frac(d: dict[str, int]) -> str:
    return f"{d['n']}/{d['N']} ({d['n'] / d['N']:.0%})" if d['N'] else 'n/a'


def print_report_tables(summary, traces, reviews) -> None:
    from agent_metrics import aggregate
    valid = [t for t in traces if t['status'] != 'error']
    complete, missing = labels_complete(traces, reviews)
    cases = [{'id': t['query_id'], 'answerable': t['answerable']} for t in traces]
    metrics = aggregate(cases, traces, reviews) if complete and reviews else None

    tool_use = {'n': sum(t.get('n_searches', 0) > 0 for t in valid), 'N': len(valid)}
    answered = [t for t in valid if t['status'] == 'answered']
    coverage = {'n': sum(bool(t['citations']) for t in answered), 'N': len(answered)}
    ans = [t for t in valid if t['answerable']]
    false_abs = {'n': sum(t['status'] == 'abstain' for t in ans), 'N': len(ans)}
    n_search = sum(t.get('n_searches', 0) for t in valid)
    n_calls = sum(t.get('n_model_calls', 0) for t in valid)
    calls = (f"{n_search} search / {len(valid)} câu (TB {n_search / len(valid):.2f}); {n_calls} lượt model"
             if valid else 'n/a')
    if metrics:
        g = metrics['grounded_answer']
        grounded = f"0:{g['0']} 1:{g['1']} 2:{g['2']} (N={g['N']}), TB {g['mean']:.2f}" if g['N'] else 'n/a'
        cit_valid, abstention = frac(metrics['citation_validity']), frac(metrics['abstention'])
    else:
        grounded = cit_valid = abstention = PENDING
    cfg = summary['config']
    name = f"{cfg['method']}{'+filter' if cfg['use_filter'] else ''}, k={cfg['top_k']}, budget={cfg['budget']}, " \
           f"max_search={cfg['max_searches']} / {summary['split']}"

    print('\n## Bảng tổng hợp agent (dán vào báo cáo)\n')
    print('| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ '
          '| Câu có citation | Abstention đúng | Từ chối nhầm |')
    print('|---|---|---|---|---|---|---|---|')
    print(f'| {name} | {frac(tool_use)} | {calls} | {grounded} | {cit_valid} | {frac(coverage)} '
          f'| {abstention} | {frac(false_abs)} |')

    by_id = {r.get('query_id'): r for r in reviews or []}
    print('\n| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |')
    print('|---|---|---|---|---|')
    for t in traces:
        tools = []
        for e in t['events']:
            if e.get('event') == 'search':
                pages = ', '.join(f"p{r['page']}{'*' if r.get('hit') else ''}" for r in e['results']) or 'không có'
                tools.append(f"search(\"{short(e['query'], 60)}\") → {pages}; ctx {len(e['context_ids'])} chunk")
            elif e.get('event') != 'answer':
                tools.append(e['event'])
        if t['status'] == 'error':
            reply = f"lỗi: {short(t['error'], 80)}"
        else:
            reply = f"{t['status']}: {short(t['answer'] or '', 140)} — cite: {', '.join(t['citations']) or '[]'}"
        r = by_id.get(t['query_id'])
        if t['status'] == 'error':
            label = 'loại khỏi mẫu (lỗi provider)'
        elif t['query_id'] in missing or not r:
            label = PENDING
        else:
            head = (f"grade {r['answer_grade']}" if t['answerable']
                    else f"abstention {'đúng' if r['abstention_correct'] else 'sai'}")
            support = f"support {sum(r['citation_support'])}/{len(r['citation_support'])}"
            label = f"{head}, {support}" + (f"; {r['reason']}" if r.get('reason') else '')
        kind = (t.get('type') or '-') + ('' if t['answerable'] else ' (thiếu evidence)')
        cell = lambda s: str(s).replace('|', '\\|').replace('\n', ' ')  # noqa: E731
        print(f"| {t['query_id']} | {cell(kind)} | {cell('<br>'.join(tools) or 'không gọi tool')} "
              f"| {cell(reply)} | {cell(label)} |")

    u = summary['usage']
    print(f"\nTrace: {summary['output']}/traces.json | model {cfg['model']} | {u['requests']} request, "
          f"{u['prompt_tokens']}+{u['completion_tokens']} token | chi phí ${u['cost_usd'] + u['embedding_cost_usd']:.6f} "
          f"(chat ${u['cost_usd']:.6f} + embedding ${u['embedding_cost_usd']:.6f}) | "
          f"median {summary['median_s']:.1f}s/câu, tổng {summary['total_s']:.1f}s")
    print('(* = chunk chứa evidence gold, chấm tự động; không thay cho nhãn thủ công)')
    errors = [t['query_id'] for t in traces if t['status'] == 'error']
    if errors:
        print(f"Lỗi provider (loại khỏi mẫu): {', '.join(errors)}")
    if missing:
        print(f"Chưa chấm đủ nhãn trong reviews.json: {', '.join(missing)}")


def show(folder: Path, args) -> None:
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    traces = json.loads((folder / 'traces.json').read_text(encoding='utf-8'))
    reviews_path = folder / 'reviews.json'
    reviews = json.loads(reviews_path.read_text(encoding='utf-8')) if reviews_path.is_file() else None
    if not args.quiet:
        for t in traces:
            print_trace(t, args.width)
    print_report_tables(summary, traces, reviews)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', choices=('dev', 'test'))
    p.add_argument('--show', type=Path, metavar='FOLDER', help='reprint a saved run instead of running the agent')
    p.add_argument('--method', default='dense', choices=('dense', 'lexical', 'hybrid'))
    p.add_argument('--use-filter', action='store_true', help="pass each question's metadata filter to search")
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--budget', type=int, default=800, help='context budget in characters')
    p.add_argument('--max-searches', type=int, default=4)
    p.add_argument('--model', default=None, help='chat model; default CHAT_MODEL or openai/gpt-4o-mini')
    p.add_argument('--strategy', default='B', help='Day 6 index for dense retrieval')
    p.add_argument('--pool', type=int, default=20, help='hybrid: candidates per list before RRF')
    p.add_argument('--rrf-k', type=int, default=60)
    p.add_argument('--only', default=None, help='comma-separated query IDs to run')
    p.add_argument('--questions', type=Path, default=DEFAULT_DATA / 'questions.json')
    p.add_argument('--chunks', type=Path, default=DEFAULT_DATA / 'chunks.json')
    p.add_argument('--benchmark', type=Path, default=ROOT / 'benchmark' / 'benchmark.json')
    p.add_argument('--output', type=Path, default=None, help='default: starter/results/agent-<split>[-method][-filter]')
    p.add_argument('--quiet', action='store_true', help='print only the report tables, no step-by-step trace')
    p.add_argument('--width', type=int, default=110, help='terminal text width')
    args = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    if args.show:
        return show(args.show, args)
    if not args.split:
        p.error('--split is required unless --show is given')
    if min(args.top_k, args.budget, args.max_searches) < 1:
        p.error('--top-k, --budget and --max-searches must be >= 1')
    args.output = (args.output or default_output(args)).resolve()

    load_env()
    from agentic import Controller
    cases = load_cases(args.questions, args.benchmark, args.split)
    if args.only:
        wanted = {i.strip() for i in args.only.split(',') if i.strip()}
        unknown = wanted - {c['id'] for c in cases}
        if unknown:
            p.error(f"unknown IDs for split {args.split}: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c['id'] in wanted]
    controller = Controller(args.model)
    retriever = build_retriever(args)

    traces = []
    started = time.perf_counter()
    for i, case in enumerate(cases, start=1):
        print(f'[{i}/{len(cases)}] {case["id"]} ...', file=sys.stderr, flush=True)
        t = run_case(case, controller, retriever, args)
        traces.append(t)
        if not args.quiet:
            print_trace(t, args.width)

    usage = {**controller.usage, 'embedding_cost_usd': float(getattr(retriever, 'cost_usd', 0.0))}
    summary = {'split': args.split, 'n': len(traces), 'output': args.output.relative_to(ROOT).as_posix()
               if args.output.resolve().is_relative_to(ROOT) else str(args.output),
               'status': {s: sum(t['status'] == s for t in traces) for s in ('answered', 'abstain', 'error')},
               'config': {'method': args.method, 'use_filter': args.use_filter, 'top_k': args.top_k,
                          'budget': args.budget, 'max_searches': args.max_searches, 'model': controller.model,
                          'day6_index': args.strategy if args.method != 'lexical' else None,
                          'only': args.only},
               'usage': usage, 'total_s': time.perf_counter() - started,
               'median_s': statistics.median(t['elapsed_ms'] for t in traces) / 1000 if traces else 0.0}

    args.output.mkdir(parents=True, exist_ok=True)
    template = review_template(traces)
    files = {'cases.json': [{'id': c['id'], 'split': c['split'], 'answerable': c['answerable']} for c in cases],
             'traces.json': traces, 'summary.json': summary, 'reviews_template.json': template}
    reviews_path = args.output / 'reviews.json'
    if not reviews_path.is_file():
        files['reviews.json'] = template
    for name, obj in files.items():
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    reviews = json.loads(reviews_path.read_text(encoding='utf-8'))

    print_report_tables(summary, traces, reviews)
    print(f"\nSaved {', '.join(files)} to {summary['output']}")
    if 'reviews.json' not in files:
        print('reviews.json already existed and was kept; compare it with reviews_template.json '
              '(citations may differ after a re-run).')


if __name__ == '__main__':
    main()
