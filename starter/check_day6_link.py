"""Read-only check that Day 7 is wired to the Day 6 chunks / ids / sources / model / pgvector store.

Usage (from the Day 7 repo root, Day 6 Docker running):
    python starter/check_day6_link.py                 # no API cost
    python starter/check_day6_link.py --live          # + 1 real query embedding per index (tiny cost)
    python starter/check_day6_link.py --day6 <path-to-Day6-repo> --split dev

Env is read from Day 7 `.env` if present, otherwise from Day 6 `starter/.env`.
Only the chosen split's gold is inspected (default dev), so test gold stays unseen.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6 = ROOT.parents[1] / 'class3b_lab06' / 'K5-L3B-DAY06-NguyenVanAnhTrung-2A202603903-DataIngestion'
DAY6_DEFAULT_MODEL = 'openai/text-embedding-3-small'
failures: list[str] = []


def check(ok: bool, label: str, detail: str = '') -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {label}" + (f' -> {detail}' if detail else ''))
    if not ok:
        failures.append(label)


def norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def pages_of(value) -> list[int]:
    return [int(p) for p in (value if isinstance(value, list) else [value])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', type=Path, default=DEFAULT_DAY6)
    parser.add_argument('--split', default='dev', choices=['dev', 'test'])
    parser.add_argument('--strategies', default='A,B')
    parser.add_argument('--live', action='store_true', help='run one real Retriever.search per index')
    args = parser.parse_args()

    day6_env = args.day6 / 'starter' / '.env'
    env_file = ROOT / '.env' if (ROOT / '.env').is_file() else day6_env
    load_dotenv(env_file)
    print(f'env file: {env_file}')

    # 1. Same embedding model as Day 6 ingestion.
    day6_model = (dotenv_values(day6_env).get('EMBEDDING_MODEL') or '').strip() or DAY6_DEFAULT_MODEL
    day7_model = os.getenv('EMBEDDING_MODEL', '').strip() or DAY6_DEFAULT_MODEL
    check(day6_model == day7_model, 'Day 7 query model == Day 6 index model', f'{day7_model} vs {day6_model}')
    check(bool(os.getenv('DATABASE_URL')), 'DATABASE_URL set')

    # 2. Same corpus (manifest) as Day 6.
    m7 = json.loads((ROOT / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    m6 = json.loads((args.day6 / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    ids7 = {d['source_id']: Path(d['path']).name for d in m7['documents']}
    ids6 = {d['source_id']: Path(d['path']).name for d in m6['documents']}
    check(ids7 == ids6, 'Day 7 manifest source_id/source == Day 6', str(ids7))

    import psycopg
    try:
        conn = psycopg.connect(os.environ['DATABASE_URL'], connect_timeout=5)
    except (psycopg.Error, KeyError) as error:
        sys.exit(f'[FAIL] cannot reach Day 6 store ({type(error).__name__}); start Day 6 Docker')

    strategies = args.strategies.split(',')
    bench = json.loads((ROOT / 'benchmark' / 'benchmark.json').read_text(encoding='utf-8'))
    questions = [q for q in bench['questions'] if q.get('split') == args.split]

    with conn:
        dims = conn.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid='chunks'::regclass "
                            "AND attname='embedding'").fetchone()[0]
        print(f'store: chunks.embedding vector({dims})')
        for s in strategies:
            print(f'\n== index {s} ==')
            # 3. Store ids/sources are the ones in the manifest.
            rows = conn.execute('SELECT chunk_id, document_id, source, page, text FROM chunks '
                                'WHERE strategy=%s', (s,)).fetchall()
            check(bool(rows), f'index {s} has chunks', f'{len(rows)} rows')
            docs = {(r[1], r[2]) for r in rows}
            check(all(ids7.get(d) == src for d, src in docs), 'stored document_id/source match manifest', str(docs))

            # 4. Benchmark gold -> a real chunk (source_id + page + evidence) in this index.
            missing = []
            for q in questions:
                quote, src, pages = norm(q['evidence_quote']), q['gold_source_id'], pages_of(q['gold_page'])
                if not any(r[1] == src and r[3] in pages and quote in norm(r[4]) for r in rows):
                    missing.append(q['id'])
            check(not missing, f'{args.split} gold evidence found inside one stored chunk',
                  f'{len(questions) - len(missing)}/{len(questions)}' + (f' missing: {missing}' if missing else ''))

    # 5. Exported Day 7 chunks.json (if any) mirrors the store.
    converted = ROOT / 'starter' / 'data' / 'converted' / 'chunks.json'
    if converted.is_file():
        exported = {c['id']: c for c in json.loads(converted.read_text(encoding='utf-8'))}
        with psycopg.connect(os.environ['DATABASE_URL']) as c2:
            db = {r[0]: r for r in c2.execute('SELECT chunk_id, document_id, source, page, text FROM chunks '
                                              'WHERE strategy = ANY(%s)', (strategies,))}
        bad = [i for i, c in exported.items() if i not in db or
               (c['source_id'], c['source'], c['page'], c['text']) != db[i][1:]]
        check(not bad, 'converted chunks.json ids/source/page/text match store', f'{len(bad)} mismatched')
    else:
        print(f'\n[skip] no {converted.relative_to(ROOT)} (only needed for TF-IDF baseline)')

    # 6. Day 7 retriever end-to-end (real query embedding).
    if args.live:
        sys.path.insert(0, str(ROOT / 'starter'))
        from agentic import Retriever
        q = questions[0] if questions else {'question': 'What is observational data?'}
        for s in strategies:
            r = Retriever(s)
            hits = r.search(q['question'], filters=None, top_k=3)
            keys = {'id', 'source_id', 'source', 'page', 'text', 'metadata', 'score'}
            check(bool(hits) and keys <= hits[0].keys(), f'Retriever({s}).search returns Day 7 chunk format',
                  f"top1={hits[0]['id']} p{hits[0]['page']} score={hits[0]['score']:.3f}" if hits else 'no hits')

    print(f"\n{'DAY 6 -> DAY 7 LINK OK' if not failures else f'{len(failures)} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
