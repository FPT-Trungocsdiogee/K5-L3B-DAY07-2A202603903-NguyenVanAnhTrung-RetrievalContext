"""Query transformation with a real chat model (OpenRouter), plus a retriever wrapper.

Modes:
  none     -> [question]
  rewrite  -> [one keyword-focused rewrite]
  multi    -> [question, n paraphrases]
  hyde     -> [question, hypothetical answer passage]   (HyDE)

Only the question text is sent to the model; no gold, filter or evidence.
Provider errors propagate (agentic.ProviderError); there is no fake fallback.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
sys.path.insert(0, str(ROOT / 'src'))
from retrieval import SearchFn, rrf_fuse  # noqa: E402

MODES = ('none', 'rewrite', 'multi', 'hyde')

PROMPTS = {
    'rewrite': ('Rewrite the user question into ONE short search query for a textbook chapter on data '
                'collection and preparation. Keep key technical terms, add likely synonyms, drop filler words. '
                'Return JSON: {"queries": ["..."]}'),
    'multi': ('Write {n} different search queries that could find the passage answering the user question in a '
              'textbook chapter on data collection and preparation. Vary wording and terminology; do not answer. '
              'Return JSON: {{"queries": ["...", ...]}}'),
    'hyde': ('Write a short textbook-style passage (2-4 sentences) that would answer the user question, as it might '
             'appear in a data science textbook chapter on collecting and preparing data. '
             'Return JSON: {"queries": ["<passage>"]}'),
}

LLM = Callable[[str, str], tuple[str, dict[str, Any]]]  # (system, user) -> (text, usage)


def openrouter_llm(model: str | None = None, *, temperature: float = 0.0, timeout: float = 60.0) -> LLM:
    from agentic import ProviderError, _openrouter_post
    model = model or os.getenv('TRANSFORM_MODEL', '').strip() or os.getenv('CHAT_MODEL', '').strip() or 'openai/gpt-4o-mini'
    api_key = os.getenv('OPENROUTER_API_KEY', '').strip()
    if not api_key:
        raise ProviderError('OPENROUTER_API_KEY is missing')

    def call(system: str, user: str) -> tuple[str, dict[str, Any]]:
        payload = _openrouter_post('/chat/completions', {
            'model': model, 'temperature': temperature, 'usage': {'include': True},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        }, api_key=api_key, timeout=timeout)
        try:
            text = payload['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f'unexpected response: {json.dumps(payload)[:300]}') from error
        return text, payload.get('usage') or {}

    call.model = model  # type: ignore[attr-defined]
    return call


def parse_queries(text: str) -> list[str]:
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.S)
        if not match:
            raise ValueError(f'model output is not JSON: {text[:200]!r}')
        data = json.loads(match.group(0))
    queries = data.get('queries') if isinstance(data, dict) else None
    if not isinstance(queries, list):
        raise ValueError(f'model output has no "queries" list: {text[:200]!r}')
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()]


class QueryTransformer:
    """Turns a question into the list of queries to run. Results are cached per (mode, question)."""

    def __init__(self, mode: str = 'none', *, llm: LLM | None = None, n: int = 3,
                 cache_path: Path | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode, self.n, self.cache_path = mode, n, cache_path
        self.llm = llm if llm is not None or mode == 'none' else openrouter_llm()
        self.usage = {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cost_usd': 0.0}
        self.cache: dict[str, list[str]] = {}
        if cache_path and cache_path.is_file():
            self.cache = json.loads(cache_path.read_text(encoding='utf-8'))

    def __call__(self, question: str) -> list[str]:
        if self.mode == 'none':
            return [question]
        key = f'{self.mode}:{self.n}:{question}'
        if key not in self.cache:
            system = PROMPTS[self.mode].format(n=self.n) if self.mode == 'multi' else PROMPTS[self.mode]
            text, usage = self.llm(system, question)
            self.usage['requests'] += 1
            self.usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
            self.usage['completion_tokens'] += usage.get('completion_tokens', 0)
            self.usage['cost_usd'] += float(usage.get('cost') or 0)
            generated = parse_queries(text)
            if not generated:
                raise ValueError(f'{self.mode}: model returned no query for {question!r}')
            if self.mode == 'rewrite':
                queries = generated[:1]
            elif self.mode == 'multi':
                queries = [question] + generated[:self.n]
            else:  # hyde
                queries = [question, generated[0]]
            self.cache[key] = list(dict.fromkeys(queries))  # dedupe, keep order
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding='utf-8')
        return self.cache[key]


class TransformedRetriever:
    """Runs every transformed query on the base retriever and fuses the lists with RRF."""

    def __init__(self, base: SearchFn, transformer: QueryTransformer, *, pool: int = 20, k: int = 60) -> None:
        self.base, self.transformer, self.pool, self.k = base, transformer, pool, k
        self.last_queries: list[str] = []

    @property
    def cost_usd(self) -> float:
        return getattr(self.base, 'cost_usd', 0.0) + self.transformer.usage['cost_usd']

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        self.last_queries = self.transformer(query)
        if len(self.last_queries) == 1:
            return self.base.search(self.last_queries[0], filters=filters, top_k=top_k)
        pool = max(self.pool, top_k)
        lists = [self.base.search(q, filters=filters, top_k=pool) for q in self.last_queries]
        fused = rrf_fuse(lists, k=self.k, labels=[f'q{i}' for i in range(len(lists))])
        return fused[:top_k]
