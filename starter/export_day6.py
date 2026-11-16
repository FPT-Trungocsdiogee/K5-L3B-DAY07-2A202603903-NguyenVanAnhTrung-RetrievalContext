"""Export Day 6 pgvector rows (+ benchmark questions) to JSON arrays for convert_day6.py.

Usage (from the Day 7 repo root, Day 6 Docker running):
    python starter/export_day6.py                     # index B -> starter/data/day6/
    python starter/export_day6.py --strategy A --output starter/data/day6-A

Then:
    python starter/convert_day6.py --chunks starter/data/day6/chunks.json \
        --questions starter/data/day6/questions.json --output starter/data/converted

Read-only on the database. Embeddings are not exported (TF-IDF does not need them;
dense retrieval queries the store directly through Retriever).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6 = ROOT.parents[1] / 'class3b_lab06' / 'K5-L3B-DAY06-NguyenVanAnhTrung-2A202603903-DataIngestion'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='B', help='Day 6 index to export (A or B)')
    parser.add_argument('--day6', type=Path, default=DEFAULT_DAY6)
    parser.add_argument('--output', type=Path, default=ROOT / 'starter' / 'data' / 'day6')
    args = parser.parse_args()

    load_dotenv(ROOT / '.env' if (ROOT / '.env').is_file() else args.day6 / 'starter' / '.env')
    url = os.getenv('DATABASE_URL', '').strip()
    if not url:
        sys.exit('DATABASE_URL missing (Day 7 .env or Day 6 starter/.env)')

    import psycopg
    with psycopg.connect(url, connect_timeout=5) as conn:
        rows = conn.execute('SELECT chunk_id, document_id, source, page, text, metadata FROM chunks '
                            'WHERE strategy = %s ORDER BY document_id, page, chunk_id',
                            (args.strategy,)).fetchall()
    if not rows:
        sys.exit(f"no rows for strategy '{args.strategy}'")
    chunks = [{'chunk_id': r[0], 'document_id': r[1], 'source': r[2], 'page': r[3], 'text': r[4],
               'metadata': {**(r[5] or {}), 'day6_strategy': args.strategy}} for r in rows]

    # convert_day6 expects gold_page as int (baseline compares chunk page == gold page);
    # a list of pages becomes one gold entry per page.
    # Day 6 matched evidence ignoring case/whitespace, but the Day 7 baseline uses an exact
    # substring test, and PDF text keeps line breaks. So each quote is replaced by the exact
    # span found in a chunk on the gold page (chunk text itself is never modified).
    def exact_span(source_id: str, page: int, quote: str) -> str | None:
        pattern = re.compile(r'\s+'.join(map(re.escape, quote.split())), re.IGNORECASE)
        for c in chunks:
            if c['document_id'] == source_id and c['page'] == page:
                match = pattern.search(c['text'])
                if match:
                    return match.group(0)
        return None

    bench = json.loads((ROOT / 'benchmark' / 'benchmark.json').read_text(encoding='utf-8'))
    questions = []
    for q in bench['questions']:
        q = dict(q)
        pages = q['gold_page'] if isinstance(q['gold_page'], list) else [q['gold_page']]
        gold = [{'source_id': q['gold_source_id'], 'page': int(p), 'evidence': q['evidence_quote']} for p in pages]
        gold += [{'source_id': g['gold_source_id'], 'page': int(g['gold_page']), 'evidence': g['evidence_quote']}
                 for g in q.get('gold_alternatives', [])]
        found = []
        for g in gold:
            span = exact_span(g['source_id'], g['page'], g['evidence'])
            if span:
                found.append({**g, 'evidence': span})
        if found:
            gold = found      # drop listed pages that do not hold the quote (e.g. q05 [72,73,74])
        else:
            print(f"[warn] {q['id']}: evidence not found whole in any index {args.strategy} chunk")
        q['gold'] = gold
        questions.append(q)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (('chunks.json', chunks), ('questions.json', questions)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {len(chunks)} chunks (index {args.strategy}) and {len(questions)} questions -> {args.output}')


if __name__ == '__main__':
    main()
