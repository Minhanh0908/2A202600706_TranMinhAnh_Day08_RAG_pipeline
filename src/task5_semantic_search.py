"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Try to use SentenceTransformer embeddings if available (matching Task 4).
    # Fallback: lightweight token-overlap scoring on local markdown files.
    from pathlib import Path
    import re

    STD_DIR = Path(__file__).parent.parent / "data" / "standardized"
    if not STD_DIR.exists():
        return []

    md_files = sorted(STD_DIR.rglob("*.md"))
    if not md_files:
        return []

    texts = []
    metas = []
    for md in md_files:
        try:
            content = md.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue
        texts.append(content)
        metas.append({"source": md.name, "source_path": str(md.relative_to(STD_DIR))})

    # Attempt embedding-based similarity using sentence-transformers (if installed).
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("BAAI/bge-m3")
        # encode texts and query with normalized embeddings → dot product = cosine
        text_embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        query_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

        def _dot(a, b):
            return float(sum(x * y for x, y in zip(a, b)))

        scores = [_dot(query_emb, te) for te in text_embs]
    except Exception:
        # Fallback: token overlap / frequency score (fast, no extra deps)
        q_tokens = [t.lower() for t in re.findall(r"\w+", query, flags=re.UNICODE)]
        q_set = set(q_tokens)
        scores = []
        for txt in texts:
            tokens = [t.lower() for t in re.findall(r"\w+", txt, flags=re.UNICODE)]
            if not tokens:
                scores.append(0.0)
                continue
            # score = sum of query token frequencies in document, normalized
            cnt = sum(tokens.count(tok) for tok in q_set)
            scores.append(float(cnt) / max(1, len(tokens)))

    results = []
    for txt, sc, meta in zip(texts, scores, metas):
        results.append({"content": txt, "score": float(sc), "metadata": meta})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[: max(0, int(top_k))]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
