import pytest
import torch

from app.ml import ml_engine


class DummyEncodeModel:
    def encode(self, data, convert_to_tensor=True):
        if isinstance(data, list):
            return torch.stack([
                torch.tensor([float(len(str(item))), 0.0], dtype=torch.float32)
                for item in data
            ])
        return torch.tensor([1.0, 0.0], dtype=torch.float32)


class DummyQueryModel:
    def encode(self, data, convert_to_tensor=True):
        return torch.tensor([1.0, 0.0], dtype=torch.float32)


def test_compute_embeddings_returns_tensor_shape():
    texts = ["one", "two"]
    result = ml_engine.compute_embeddings(texts, DummyEncodeModel())

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 2)


def test_encode_query_returns_tensor_for_query():
    result = ml_engine.encode_query(DummyQueryModel(), "query")

    assert isinstance(result, torch.Tensor)
    assert result.shape == (2,)


def test_compute_batch_scores_returns_expected_similarity():
    query_embedding = torch.tensor([1.0, 0.0], dtype=torch.float32)
    batch_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    scores = ml_engine.compute_batch_scores(query_embedding, batch_embeddings)

    assert scores.shape == (2,)
    assert scores[0].item() == pytest.approx(1.0)
    assert scores[1].item() == pytest.approx(0.0)


def test_select_top_k_returns_highest_scores():
    scores = torch.tensor([0.2, 0.8, 0.5], dtype=torch.float32)
    contents = ["first", "second", "third"]
    top_k = ml_engine.select_top_k([], scores, contents, 2)

    sorted_top_k = sorted(top_k, reverse=True)
    assert sorted_top_k[0][0] == pytest.approx(0.8)
    assert sorted_top_k[0][1] == "second"
    assert sorted_top_k[1][0] == pytest.approx(0.5)
    assert sorted_top_k[1][1] == "third"


def test_search_similar_texts_selects_top_documents():
    corpus_texts = ["a", "b", "c"]
    corpus_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=torch.float32
    )
    results = ml_engine.search_similar_texts(
        "query",
        corpus_texts,
        corpus_embeddings,
        DummyQueryModel(),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][1] >= results[1][1]
