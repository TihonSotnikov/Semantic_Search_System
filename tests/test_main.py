import torch

from app.main import embeddings_from_docs


class StubDocumentModel:
    def encode(self, texts, convert_to_tensor=True):
        assert texts == ["# Title\nBody content"]
        return torch.tensor([[0.1, 0.2]], dtype=torch.float32)


def test_embeddings_from_docs_builds_knowledge_records():
    documents = [{"title": "Title", "text": "Body content"}]
    records = embeddings_from_docs(documents, StubDocumentModel())

    assert len(records) == 1
    assert records[0].title == "Title"
    assert records[0].text == "Body content"
    assert isinstance(records[0].vector, torch.Tensor)
