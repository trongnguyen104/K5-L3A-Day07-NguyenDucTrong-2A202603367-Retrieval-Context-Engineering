"""Kiểm tra offline cho pipeline src/: BM25, dense in-memory, hybrid RRF, metrics.

Không gọi mạng và không dùng vector ngẫu nhiên: embedder trong test là ánh xạ
đếm từ khoá xác định, chỉ để kiểm tra logic xếp hạng.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunks import check_gold_against_chunks, load_questions
from src.context import pack_context
from src.evaluate import run_split
from src.retrieval import validate_filters
from src.retrieval.bm25 import BM25Retriever, tokenize
from src.retrieval.dense import DenseRetriever, MemoryVectorStore, cosine_similarity
from src.retrieval.hybrid import HybridRRF


def chunk(chunk_id, text, page=1, source_id='doc'):
    return {'id': chunk_id, 'text': text, 'source_id': source_id,
            'source': f'{source_id}.pdf', 'page': page, 'metadata': {'year': 2025}}


CHUNKS = [
    chunk('a', 'Yêu cầu hoàn tiền được xử lý trong bảy ngày làm việc.', 1),
    chunk('b', 'Giao hàng tiêu chuẩn mất ba ngày.', 2),
    chunk('c', 'Bảo hành thiết bị mười hai tháng.', 3),
]


class KeywordEmbedder:
    """Embedding xác định theo số lần xuất hiện từ khoá; chỉ dùng trong test."""

    name = 'test-keyword'
    dimension = 3
    words = ('hoàn', 'giao', 'bảo')

    def _vector(self, text):
        lowered = text.lower()
        return [float(lowered.count(word)) for word in self.words]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FixedRetriever:
    def __init__(self, name, order):
        self.name = name
        self.order = order

    def search(self, query, *, top_k=3, filters=None):
        return [
            {**chunk(item, f'nội dung {item}'), 'score': 1.0 / (rank + 1), 'rank': rank + 1}
            for rank, item in enumerate(self.order[:top_k])
        ]


class TokenizerTests(unittest.TestCase):
    def test_fold_accents_and_case(self):
        self.assertEqual(tokenize('Hoàn tiền'), ['hoàn', 'tiền'])
        self.assertEqual(tokenize('HOÀN TIỀN', fold_accents=True), ['hoan', 'tien'])
        self.assertEqual(tokenize('Đổi trả', fold_accents=True), ['doi', 'tra'])

    def test_validate_filters(self):
        self.assertIsNone(validate_filters(None))
        self.assertEqual(validate_filters({'version': '2025'}), {'version': '2025'})
        with self.assertRaises(ValueError):
            validate_filters({'metadata': {}})
        with self.assertRaises(ValueError):
            validate_filters({'': 1})


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Retriever(CHUNKS)

    def test_rank_and_top_k(self):
        results = self.index.search('hoàn tiền mất bao lâu', top_k=2)
        self.assertEqual(results[0]['id'], 'a')
        self.assertEqual([r['rank'] for r in results], [1, 2])
        self.assertEqual(results[0]['method'], 'bm25')

    def test_filters_and_empty_query(self):
        results = self.index.search('ngày', top_k=5, filters={'source_id': 'doc', 'page': 2})
        self.assertEqual([r['id'] for r in results], ['b'])
        self.assertEqual(self.index.search('   ', top_k=3), [])

    def test_unknown_term_returns_nothing(self):
        self.assertEqual(self.index.search('ethereum', top_k=3), [])

    def test_metadata_filter(self):
        chunks = [chunk('a', 'hoàn tiền bảy ngày'),
                  {**chunk('b', 'hoàn tiền bảy ngày'), 'metadata': {'year': 2024}}]
        index = BM25Retriever(chunks)
        self.assertEqual([r['id'] for r in index.search('hoàn tiền', filters={'year': 2025})], ['a'])
        self.assertEqual(
            [r['id'] for r in index.search('hoàn tiền', filters={'metadata': {'year': 2024}})], ['b']
        )


class DenseTests(unittest.TestCase):
    def test_cosine_and_memory_store(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 1], [1, 0]), 2 ** -0.5)
        store = MemoryVectorStore(CHUNKS, KeywordEmbedder())
        retriever = DenseRetriever(store, KeywordEmbedder())
        results = retriever.search('hoàn tiền bao lâu', top_k=2)
        self.assertEqual(results[0]['id'], 'a')
        self.assertEqual(results[0]['rank'], 1)

    def test_filter_limits_candidates(self):
        retriever = DenseRetriever(MemoryVectorStore(CHUNKS, KeywordEmbedder()), KeywordEmbedder())
        results = retriever.search('hoàn tiền', top_k=3, filters={'page': 2})
        self.assertEqual([r['id'] for r in results], ['b'])


class HybridTests(unittest.TestCase):
    def test_rrf_orders_by_fused_rank(self):
        hybrid = HybridRRF(
            [(FixedRetriever('dense', ['a', 'b', 'c']), 1.0),
             (FixedRetriever('bm25', ['a', 'c', 'b']), 1.0)],
            k=60, candidate_k=3,
        )
        results = hybrid.search('bất kỳ', top_k=3)
        self.assertEqual([r['id'] for r in results], ['a', 'b', 'c'])
        self.assertAlmostEqual(results[0]['score'], 2 / 61)
        self.assertEqual(results[0]['ranks'], {'dense': 1, 'bm25': 1})
        self.assertEqual(results[0]['method'], 'hybrid-rrf')

    def test_weights_and_zero_weight(self):
        hybrid = HybridRRF(
            [(FixedRetriever('dense', ['a', 'b']), 2.0),
             (FixedRetriever('bm25', ['b', 'a']), 1.0)],
            k=60, candidate_k=2,
        )
        results = hybrid.search('x', top_k=2)
        self.assertEqual(results[0]['id'], 'a')
        zero = HybridRRF([(FixedRetriever('dense', ['a']), 1.0),
                          (FixedRetriever('bm25', ['b']), 0.0)], k=60, candidate_k=1)
        self.assertEqual([r['id'] for r in zero.search('x', top_k=2)], ['a'])


class ContextTests(unittest.TestCase):
    def test_budget_dedup_and_header(self):
        results = [{**CHUNKS[0], 'score': 1.0, 'rank': 1}, {**CHUNKS[0], 'score': 0.5, 'rank': 2}]
        context, ids = pack_context(results, 800)
        self.assertEqual(ids, ['a'])
        self.assertIn('[a | doc.pdf | page 1]', context)
        self.assertEqual(pack_context(results, 1), ('', []))
        with self.assertRaises(ValueError):
            pack_context(results, 0)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {'id': 'q1', 'split': 'dev', 'question': 'Hoàn tiền mất bao lâu?',
             'expected_answer': 'bảy ngày làm việc',
             'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'bảy ngày làm việc'}]},
            {'id': 'q2', 'split': 'dev', 'question': 'Bảo hành bao lâu?',
             'expected_answer': 'mười hai tháng',
             'gold': [{'source_id': 'doc', 'page': 3, 'evidence': 'mười hai tháng'}]},
            {'id': 'q3', 'split': 'dev', 'question': 'Có hỗ trợ đổi chuyến bay không?',
             'answerable': False, 'expected_behavior': 'search_and_abstain',
             'absence_reason': 'corpus test không có nội dung này'},
        ]

    def test_metrics_use_top3_and_skip_unanswerable(self):
        traces, summary = run_split(BM25Retriever(CHUNKS), self.questions, split='dev', top_k=1)
        self.assertEqual(summary['n_scored'], 2)
        self.assertEqual(summary['n_unanswerable'], 1)
        self.assertEqual(summary['hit1'], 2)
        self.assertEqual(summary['hit3'], 2)
        self.assertEqual(summary['context_evidence'], 2)
        self.assertEqual(summary['hit1_rate'], 1.0)
        self.assertEqual(len(traces), 3)
        self.assertIsNone(traces[2]['hit1'])

    def test_error_is_recorded_not_swallowed(self):
        class Broken:
            name = 'broken'

            def search(self, query, *, top_k=3, filters=None):
                raise RuntimeError('pgvector không phản hồi')

        _, summary = run_split(Broken(), self.questions, split='dev')
        self.assertEqual(summary['errors'], 3)
        self.assertEqual(summary['n_scored'], 0)
        self.assertEqual(summary['hit1_rate'], 0.0)

    def test_missing_split_raises(self):
        with self.assertRaises(ValueError):
            run_split(BM25Retriever(CHUNKS), self.questions, split='test')

    def test_gold_must_be_inside_chunk(self):
        check_gold_against_chunks(CHUNKS, [self.questions[0]])
        bad = {**self.questions[0],
               'gold': [{'source_id': 'doc', 'page': 2, 'evidence': 'bảy ngày làm việc'}]}
        with self.assertRaises(ValueError):
            check_gold_against_chunks(CHUNKS, [bad])

    def test_load_questions_validates_abstain_schema(self):
        with self.assertRaises(ValueError):
            # câu không answerable phải khai báo expected_behavior
            load_questions_from([{'id': 'x', 'split': 'dev', 'question': 'hỏi',
                                  'answerable': False}])


def load_questions_from(items):
    """Ghi tạm ra file để test loader, tránh phụ thuộc dữ liệu thật."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'q.json'
        path.write_text(json.dumps(items, ensure_ascii=False), encoding='utf-8')
        return load_questions(path)


if __name__ == '__main__':
    unittest.main()
