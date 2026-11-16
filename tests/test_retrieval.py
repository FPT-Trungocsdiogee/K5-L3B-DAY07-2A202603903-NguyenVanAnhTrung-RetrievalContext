"""Offline checks for src/ retrievers and query transformation (fakes only, no provider calls)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from query_transform import QueryTransformer, TransformedRetriever, parse_queries
from retrieval import HybridRetriever, LexicalRetriever, matches, rrf_fuse


def chunk(id, text, section='2.1', page=1):
    return {'id': id, 'source_id': 'doc', 'source': 'doc.pdf', 'page': page, 'text': text,
            'metadata': {'section': section, 'year': 2025}}


CHUNKS = [chunk('a', 'Observational data is collected by observing behavior.'),
          chunk('b', 'Surveys collect responses from participants.', '2.2', 2),
          chunk('c', 'Missing values can be imputed with the mean.', '2.3', 3)]


class FakeDense:
    def __init__(self, order): self.order = order
    def search(self, query, *, filters=None, top_k=3):
        return [{**c, 'score': 1.0} for c in CHUNKS if c['id'] in self.order and matches(c, filters)][:top_k]


class RetrievalTests(unittest.TestCase):
    def test_bm25_rank_and_filter(self):
        bm25 = LexicalRetriever(CHUNKS)
        self.assertEqual(bm25.search('how is observational data collected', top_k=1)[0]['id'], 'a')
        self.assertEqual([r['id'] for r in bm25.search('data collected by surveys', filters={'section': '2.2'}, top_k=3)], ['b'])
        self.assertTrue(matches(CHUNKS[0], {'year': [2024, 2025], 'page': 1}))
        self.assertFalse(matches(CHUNKS[0], {'unknown': 'x'}))

    def test_rrf_and_hybrid(self):
        fused = rrf_fuse([[{'id': 'x'}, {'id': 'y'}], [{'id': 'y'}, {'id': 'z'}]], labels=['d', 'l'])
        self.assertEqual(fused[0]['id'], 'y')
        self.assertEqual(fused[0]['ranks'], {'d': 2, 'l': 1})
        hybrid = HybridRetriever(FakeDense(['c', 'b']), LexicalRetriever(CHUNKS), pool=5)
        results = hybrid.search('surveys responses', top_k=2)
        self.assertEqual(results[0]['id'], 'b')
        self.assertTrue(all(r['strategy'] == 'hybrid' for r in results))

    def test_transform_modes(self):
        llm = lambda system, user: ('```json\n{"queries": ["survey responses", "questionnaire"]}\n```',
                                    {'prompt_tokens': 3, 'completion_tokens': 2, 'cost': 0.001})
        multi = QueryTransformer('multi', llm=llm, n=2)
        self.assertEqual(multi('q?'), ['q?', 'survey responses', 'questionnaire'])
        multi('q?')  # cached: no second call
        self.assertEqual(multi.usage['requests'], 1)
        self.assertEqual(QueryTransformer('rewrite', llm=llm)('q?'), ['survey responses'])
        self.assertEqual(QueryTransformer('hyde', llm=llm)('q?'), ['q?', 'survey responses'])
        self.assertEqual(QueryTransformer('none')('q?'), ['q?'])
        retriever = TransformedRetriever(LexicalRetriever(CHUNKS), QueryTransformer('rewrite', llm=llm))
        self.assertEqual(retriever.search('unrelated words', top_k=1)[0]['id'], 'b')
        with self.assertRaises(ValueError):
            parse_queries('no json here')


class AgentCaseTests(unittest.TestCase):
    def test_absent_cases_split_and_scores(self):
        import json, tempfile
        from types import SimpleNamespace
        from run_retrieval import absent_summary, evaluate_absent, load_agent_cases
        cases = {'questions': [], 'agent_cases': [
            {'id': 'a1', 'split': 'dev', 'question': 'random forest hyperparameters?', 'answerable': False,
             'absence_reason': 'secret'},
            {'id': 'a2', 'split': 'test', 'question': 'p-value?', 'answerable': False}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'b.json'
            path.write_text(json.dumps(cases), encoding='utf-8')
            dev = load_agent_cases(path, 'dev')
        self.assertEqual([c['id'] for c in dev], ['a1'])
        args = SimpleNamespace(use_filter=False, top_k=2, budget=800)
        traces = evaluate_absent(LexicalRetriever(CHUNKS), dev, args, 'bm25')
        self.assertEqual(traces[0]['answerable'], False)
        self.assertIsNone(traces[0]['top1_score'])  # no word overlap -> no results
        answerable = [{'results': [{'score': 0.5}]}]
        self.assertEqual(absent_summary(answerable, traces)['empty_results'], 1)


if __name__ == '__main__':
    unittest.main()
