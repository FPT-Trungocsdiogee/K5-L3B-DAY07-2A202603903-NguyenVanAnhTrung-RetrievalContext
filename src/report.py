"""Plain-text terminal output for retrieval runs (ASCII only, safe on any console)."""
from typing import Any, Sequence

YES, NO = 'x', '.'


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: str | None = None) -> str:
    """Render rows as an aligned text table. align: one char per column, 'l' or 'r'."""
    cells = [[str(h) for h in headers]] + [[str(v) for v in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    align = align or 'l' * len(headers)

    def line(row):
        return '  '.join(v.rjust(w) if a == 'r' else v.ljust(w) for v, w, a in zip(row, widths, align)).rstrip()

    sep = '  '.join('-' * w for w in widths)
    return '\n'.join([line(cells[0]), sep] + [line(r) for r in cells[1:]])


def ratio(count: int, n: int) -> str:
    return f'{count}/{n} ({count / n:.0%})' if n else 'n/a'


def short(text: str, width: int) -> str:
    text = ' '.join(str(text).split())
    return text if len(text) <= width else text[:width - 3] + '...'


def describe_config(summary: dict[str, Any]) -> str:
    cfg = summary.get('config', {})
    parts = [f"top_k={cfg.get('top_k')}", f"budget={cfg.get('budget_chars')}",
             f"filter={'on' if cfg.get('use_filter') else 'off'}"]
    method = summary.get('method')
    if method in ('dense', 'hybrid'):
        parts.append(f"index={cfg.get('day6_index')}")
    if method == 'hybrid' or summary.get('transform', 'none') != 'none':
        parts.append(f"pool={cfg.get('pool')} rrf_k={cfg.get('rrf_k')}")
    if method == 'hybrid':
        w = cfg.get('weights', {})
        parts.append(f"w_dense={w.get('dense')} w_lex={w.get('lexical')}")
    if method in ('lexical', 'hybrid') and cfg.get('bm25'):
        parts.append(f"bm25 k1={cfg['bm25'].get('k1')} b={cfg['bm25'].get('b')}")
    if cfg.get('transform_model'):
        parts.append(f"transform_model={cfg['transform_model']}")
    return ', '.join(parts)


def print_questions(traces: Sequence[dict[str, Any]], *, show_queries: bool = False) -> None:
    rows = []
    for t in traces:
        top = ', '.join(f"p{r['page']}{'*' if r.get('hit') else ''}" for r in t['results']) or '(none)'
        gold = ','.join(sorted({f"p{g['page']}" for g in t['gold']}))
        rows.append([t['query_id'], YES if t['hit1'] else NO, YES if t['hit3'] else NO,
                     YES if t['context_hit'] else NO, gold, top, len(t['context_ids']),
                     f"{t['elapsed_ms']:.0f}", short(t['question'], 48)])
    print(table(['id', 'H@1', 'H@k', 'Ctx', 'gold', 'top-k pages (* = evidence)', 'in ctx', 'ms', 'question'],
                rows, 'lcccllrrl'))
    if show_queries:
        for t in traces:
            queries = t.get('queries') or []
            if len(queries) > 1 or (queries and queries[0] != t['question']):
                print(f"\n{t['query_id']} queries:")
                for q in queries:
                    print(f'  - {short(q, 110)}')


def print_misses(traces: Sequence[dict[str, Any]], *, width: int = 110) -> None:
    misses = [t for t in traces if not t['context_hit']]
    if not misses:
        print('No misses: evidence reached the context for every question.')
        return
    for t in misses:
        print(f"\n{t['query_id']}  {short(t['question'], width)}")
        for g in t['gold']:
            print(f"  gold   p{g['page']}: {short(g['evidence'], width - 12)}")
        for rank, r in enumerate(t['results'], start=1):
            flag = 'EVIDENCE' if r.get('hit') else ''
            in_ctx = 'ctx' if r['id'] in t['context_ids'] else '   '
            print(f"  #{rank} {in_ctx} p{r['page']} score={r['score']:.4f} {flag}")
            print(f"         {short(r['text'], width - 9)}")


def print_summary(summary: dict[str, Any]) -> None:
    n = summary['n']
    print(f"\n{summary['strategy']} | split={summary['split']} | n={n}")
    print(f'  config:  {describe_config(summary)}')
    print(f"  Hit@1:                  {ratio(summary['hit1'], n)}")
    print(f"  Evidence Hit@k:         {ratio(summary['hit3'], n)}")
    print(f"  Context Evidence Rate:  {ratio(summary['context_hit'], n)}")
    print(f"  avg context chars:      {summary['avg_context_chars']:.0f}")
    print(f"  median latency:         {summary['median_ms']:.1f} ms   (index build {summary['index_seconds']:.2f} s)")
    print(f"  service cost:           ${summary.get('service_cost_usd', 0):.6f}")
    usage = summary.get('transform_usage')
    if usage:
        print(f"  transform calls:        {usage['requests']} "
              f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens)")


def print_absent(traces: Sequence[dict[str, Any]], *, text: bool = False, show_queries: bool = False,
                 width: int = 110) -> None:
    """Unanswerable agent_cases: no gold, so show what retrieval returned and the top-1 score."""
    rows = [[t['query_id'], f"{t['top1_score']:.4f}" if t['top1_score'] is not None else '-',
             ', '.join(f"p{r['page']}" for r in t['results']) or '(none)', len(t['context_ids']),
             f"{t['elapsed_ms']:.0f}", short(t['question'], 60)] for t in traces]
    print(table(['id', 'top1 score', 'top-k pages', 'in ctx', 'ms', 'question'], rows, 'lrlrrl'))
    for t in traces:
        queries = t.get('queries') or []
        if show_queries and (len(queries) > 1 or (queries and queries[0] != t['question'])):
            print(f"\n{t['query_id']} queries:")
            for q in queries:
                print(f'  - {short(q, width)}')
        if text:
            print(f"\n{t['query_id']}  {short(t['question'], width)}")
            for rank, r in enumerate(t['results'], start=1):
                in_ctx = 'ctx' if r['id'] in t['context_ids'] else '   '
                print(f"  #{rank} {in_ctx} p{r['page']} score={r['score']:.4f}")
                print(f"         {short(r['text'], width - 9)}")


def print_absent_summary(absent: dict[str, Any] | None) -> None:
    if not absent:
        return
    fmt = lambda v: '-' if v is None else f'{v:.4f}'
    print(f"  agent_cases (no evidence): n={absent['n']}, empty results={absent['empty_results']}")
    print(f"    top-1 score  absent: median {fmt(absent['top1_score_median'])}, max {fmt(absent['top1_score_max'])}"
          f"  | answerable: median {fmt(absent['answerable_top1_score_median'])},"
          f" min {fmt(absent['answerable_top1_score_min'])}")
    if absent['below_answerable_min'] is not None:
        print(f"    below the lowest answerable top-1 score: {absent['below_answerable_min']}/{absent['n']}"
              ' (a score threshold alone could reject these)')


def print_run(summary: dict[str, Any], traces: Sequence[dict[str, Any]], *,
              details: bool = True, misses: bool = False, show_queries: bool = False,
              absent_traces: Sequence[dict[str, Any]] | None = None) -> None:
    if details:
        print_questions(traces, show_queries=show_queries)
    if misses:
        print('\n== Misses (evidence not in context) ==')
        print_misses(traces)
    if absent_traces and (details or misses):
        print('\n== agent_cases without evidence (answerable=false) ==')
        print_absent(absent_traces, text=misses, show_queries=show_queries)
    print_summary(summary)
    print_absent_summary(summary.get('absent'))


def print_comparison(runs: Sequence[tuple[str, dict[str, Any]]]) -> None:
    rows = []
    for name, s in runs:
        n = s['n']
        absent = s.get('absent')
        below = (f"{absent['below_answerable_min']}/{absent['n']}"
                 if absent and absent.get('below_answerable_min') is not None else '-')
        rows.append([name, s['strategy'], s['split'], ratio(s['hit1'], n), ratio(s['hit3'], n),
                     ratio(s['context_hit'], n), below, f"{s['median_ms']:.1f}", f"{s.get('service_cost_usd', 0):.6f}",
                     describe_config(s) if 'config' in s else f"top_k={s.get('top_k', 3)}"])
    print(table(['run', 'strategy', 'split', 'Hit@1', 'Evidence Hit@k', 'Context Evidence', 'absent<min',
                 'median ms', 'cost $', 'config'], rows, 'lllrrrrrrl'))
    if any(s.get('absent') for _, s in runs):
        print('\nabsent<min: agent_cases whose top-1 score is below the lowest answerable top-1 score of the same run.')
