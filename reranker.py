from sentence_transformers import (
    CrossEncoder
)

_reranker_model = None


def _get_reranker_model():
    """Lazy load the reranker model on first use."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _reranker_model


def rerank_documents(
    query,
    documents,
    top_k=12
):

    if not documents:

        return []

    pairs = [
        (
            query,
            doc.page_content
        )
        for doc in documents
    ]

    model = _get_reranker_model()
    scores = model.predict(pairs)

    ranked_docs = sorted(
        zip(
            documents,
            scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, _
        in ranked_docs[:top_k]
    ]
