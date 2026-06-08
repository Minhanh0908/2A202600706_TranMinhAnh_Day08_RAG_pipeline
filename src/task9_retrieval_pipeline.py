"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: retrieve from dense + sparse (use larger pool to allow rerank)
    dense_results = []
    sparse_results = []
    try:
        dense_results = semantic_search(query, top_k=top_k * 2) or []
    except Exception:
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=top_k * 2) or []
    except Exception:
        sparse_results = []

    # Step 2: merge by RRF
    try:
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    except Exception:
        # fallback: simple concatenation + dedup by content
        seen = set()
        merged = []
        for lst in (dense_results + sparse_results):
            cid = str(lst.get("content", ""))
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(lst.copy())
        merged = merged[: top_k * 2]

    for item in merged:
        item.setdefault("metadata", {})
        item["source"] = "hybrid"

    # Step 3: rerank (try preferred method, with graceful fallback)
    final_results: list[dict] = []
    if use_reranking and merged:
        try:
            # Try unified rerank API; for RRF we pass ranked_lists for best effect
            final_results = rerank(
                query,
                merged,
                top_k=top_k,
                method=RERANK_METHOD,
                ranked_lists=[dense_results, sparse_results],
            )
        except Exception:
            try:
                # Fallback to RRF merge sorting
                final_results = rerank(query, merged, top_k=top_k, method="rrf", ranked_lists=[dense_results, sparse_results])
            except Exception:
                final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # Helper to extract a numeric score for threshold check
    def _score_of(item: dict) -> float:
        for k in ("rerank_score", "mmr_score", "rrf_score", "score"):
            if k in item and item[k] is not None:
                try:
                    return float(item[k])
                except Exception:
                    continue
        return 0.0

    # Step 4: if no good hybrid result → fallback to PageIndex
    # Ensure final_results items include a source marker
    for fr in final_results:
        if isinstance(fr, dict):
            fr.setdefault("source", "hybrid")

    best_score = _score_of(final_results[0]) if final_results else 0.0
    if not final_results or best_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            return fallback
        except Exception:
            # If PageIndex also fails, return whatever hybrid produced (possibly empty)
            return final_results[:top_k]

    # Ensure returned items include required fields
    out = []
    seen_contents = set()
    for item in final_results[:top_k]:
        content = item.get("content", "")
        if content in seen_contents:
            continue
        seen_contents.add(content)
        entry = {
            "content": content,
            "score": _score_of(item),
            "metadata": item.get("metadata", {}),
            "source": item.get("source", "hybrid"),
        }
        out.append(entry)

    return out


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
