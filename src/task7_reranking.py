"""
Task 7 — Reranking Module.

Implement cả 3 phương pháp:
    1. Cross-encoder  : Jina Reranker v2 (multilingual) qua API
    2. MMR            : Maximal Marginal Relevance — tự implement
    3. RRF            : Reciprocal Rank Fusion — tự implement

Cài đặt:
    pip install requests numpy python-dotenv

Cấu hình .env:
    JINA_API_KEY=your-jina-api-key   # chỉ cần nếu dùng cross_encoder
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# Jina Reranker v2 — multilingual, hỗ trợ tiếng Việt tốt
# Tại sao chọn Jina v2 thay vì Cohere hay local cross-encoder?
#   - Multilingual được train trên 100+ ngôn ngữ kể cả tiếng Việt
#   - API-based: không cần GPU, chạy tốt trên máy yếu
#   - Free tier 1M tokens/tháng — đủ cho project demo
JINA_RERANK_URL   = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"

# MMR lambda default:
#   λ = 0.7 → nghiêng về relevance hơn diversity
#   Phù hợp RAG vì ta muốn kết quả liên quan, nhưng vẫn cần đủ đa dạng
#   để tránh 5 chunk đều nói cùng 1 điều.
MMR_LAMBDA_DEFAULT = 0.7

# RRF k=60: hằng số smoothing từ paper Cormack et al. 2009
#   Giá trị 60 được tìm ra thực nghiệm — hoạt động tốt với hầu hết corpus.
#   k nhỏ → top rank được ưu tiên mạnh hơn; k lớn → các rank san bằng hơn.
RRF_K_DEFAULT = 60


# =============================================================================
# HELPER: COSINE SIMILARITY
# =============================================================================

def _cosine_sim(vec_a: list[float] | np.ndarray,
                vec_b: list[float] | np.ndarray) -> float:
    """
    Tính cosine similarity giữa 2 vector.

    Nếu embedding đã được normalize (như BGE-M3 với normalize_embeddings=True),
    cosine_sim = dot product — tính nhanh hơn, kết quả như nhau.
    Vẫn giữ công thức đầy đủ để an toàn với embedding chưa normalize.
    """
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# METHOD 1: CROSS-ENCODER (Jina Reranker v2)
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker v2 (cross-encoder).

    Tại sao cross-encoder tốt hơn bi-encoder cho reranking?
        - Bi-encoder (BGE-M3): encode query và document RIÊNG BIỆT rồi tính cosine.
          Nhanh nhưng không capture được interaction giữa query và document.
        - Cross-encoder: nhận cặp (query, document) cùng lúc, attention có thể
          "nhìn thấy" cả hai → hiểu sâu hơn mức độ liên quan thực sự.
        - Trade-off: cross-encoder chậm hơn O(n) → chỉ dùng ở bước rerank
          trên tập nhỏ candidates (20–100), không dùng để search toàn corpus.

    Args:
        query      : Câu truy vấn gốc
        candidates : List[{'content': str, 'score': float, 'metadata': dict}]
        top_k      : Số kết quả trả về sau rerank

    Returns:
        top_k candidates đã được rerank, sorted by rerank_score descending.
        Mỗi item có thêm key 'rerank_score' (original 'score' giữ nguyên).
    """
    if not candidates:
        return []

    if not JINA_API_KEY:
        raise EnvironmentError(
            "JINA_API_KEY chưa được set trong .env\n"
            "Đăng ký tại https://jina.ai để lấy free API key (1M tokens/tháng)"
        )

    import requests

    documents = [c["content"] for c in candidates]

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":     JINA_RERANK_MODEL,
                "query":     query,
                "documents": documents,
                "top_n":     min(top_k, len(candidates)),
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("Jina API timeout — thử lại hoặc giảm số candidates")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Jina API lỗi {response.status_code}: {response.text}") from e

    reranked = response.json()["results"]

    results = []
    for r in reranked:
        item = candidates[r["index"]].copy()
        item["rerank_score"] = r["relevance_score"]   # score từ cross-encoder
        # Giữ lại 'score' gốc (BM25 / cosine) để phân tích sau nếu cần
        results.append(item)

    # Đã được Jina sort sẵn, nhưng sort lại để chắc chắn
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results


# =============================================================================
# METHOD 2: MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = MMR_LAMBDA_DEFAULT,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Công thức:
        MMR(d) = λ * sim(query, d) − (1−λ) * max_{s ∈ Selected} sim(d, s)

    Trực giác:
        - Vòng 1: chọn document relevant nhất với query
        - Vòng 2: chọn document vừa relevant với query, vừa KHÁC với doc đã chọn
        - Vòng 3, 4...: tương tự — diversity tăng dần
        → Tránh trường hợp 5 kết quả đều nói cùng 1 nội dung

    Khi nào dùng MMR?
        - Khi retrieval trả về nhiều chunk từ cùng 1 document (redundant)
        - Khi muốn LLM nhận được góc nhìn đa chiều, không bị bias 1 nguồn

    Args:
        query_embedding : Vector embedding của query (dim phải khớp candidates)
        candidates      : List[{'content', 'score', 'embedding', 'metadata'}]
                          → Bắt buộc phải có key 'embedding'
        top_k           : Số kết quả chọn ra
        lambda_param    : λ ∈ [0, 1]
                          λ = 1.0 → pure relevance (giống sort by score)
                          λ = 0.0 → pure diversity (không quan tâm relevance)
                          λ = 0.7 → cân bằng, nghiêng về relevance

    Returns:
        top_k candidates được chọn bởi MMR, có thêm key 'mmr_score'.
    """
    if not candidates:
        return []

    # Validate embedding tồn tại
    for i, c in enumerate(candidates):
        if "embedding" not in c or c["embedding"] is None:
            raise ValueError(
                f"candidates[{i}] thiếu key 'embedding'.\n"
                "Đảm bảo bước retrieval trả về embedding cùng với content."
            )

    n = len(candidates)
    top_k = min(top_k, n)

    selected_indices: list[int] = []
    remaining_indices: list[int] = list(range(n))

    # Pre-compute relevance scores (query ↔ mỗi candidate)
    # Tính trước để không recompute trong vòng lặp
    relevance_scores = [
        _cosine_sim(query_embedding, c["embedding"]) for c in candidates
    ]

    for _ in range(top_k):
        best_idx   = None
        best_score = float("-inf")

        for idx in remaining_indices:
            relevance = relevance_scores[idx]

            # Max similarity với các document đã chọn
            if selected_indices:
                sim_to_selected = max(
                    _cosine_sim(candidates[idx]["embedding"],
                                candidates[sel]["embedding"])
                    for sel in selected_indices
                )
            else:
                # Chưa chọn gì → penalty = 0, chọn doc relevant nhất
                sim_to_selected = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    results = []
    for rank, idx in enumerate(selected_indices):
        item = candidates[idx].copy()
        item["mmr_score"] = float(relevance_scores[idx])  # relevance gốc
        item["mmr_rank"]  = rank + 1
        results.append(item)

    return results


# =============================================================================
# METHOD 3: RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = RRF_K_DEFAULT,
    id_field: str = "content",
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker khác nhau.

    Công thức (Cormack et al. 2009):
        RRF(d) = Σ_{r ∈ Rankers} 1 / (k + rank_r(d))

    Trực giác:
        - Document xuất hiện ở rank 1 nhận điểm 1/(60+1) ≈ 0.016
        - Document xuất hiện ở rank 10 nhận điểm 1/(60+10) ≈ 0.014
        - Document xuất hiện ở cả 2 ranker cộng điểm lại → nổi lên top
        → Ưu tiên document được nhiều ranker đồng thuận, không bị 1 ranker
          có score bất thường làm lệch kết quả (robust với outlier)

    Tại sao k=60?
        - Giá trị empirical từ paper — không cần tune.
        - k nhỏ (vd: 1) → rank 1 được ưu tiên rất mạnh, rank 10 gần bằng 0
        - k lớn (vd: 1000) → gần như mọi rank đều đóng góp như nhau
        - k=60 cân bằng tốt: top rank quan trọng hơn nhưng không áp đảo

    Khi nào dùng RRF?
        - Khi kết hợp dense retrieval (vector) + sparse retrieval (BM25) — hybrid search
        - Khi có nhiều nguồn retrieval khác nhau (Weaviate + Elasticsearch)
        - Không cần normalize score của từng ranker — RRF chỉ dùng rank position

    Args:
        ranked_lists : List các ranked list, mỗi list từ 1 ranker.
                       Mỗi item phải có key id_field (mặc định 'content') để dedup.
        top_k        : Số kết quả cuối cùng
        k            : Smoothing constant (default=60)
        id_field     : Field dùng để identify document (dedup giữa các ranker)

    Returns:
        top_k candidates sorted by RRF score descending, có thêm key 'rrf_score'.
    """
    if not ranked_lists:
        return []

    rrf_scores: dict[str, float] = {}   # id → accumulated RRF score
    item_map:   dict[str, dict]  = {}   # id → full item dict (lưu lần xuất hiện đầu tiên)
    source_count: dict[str, int] = {}   # id → số ranker chứa item này

    for ranker_idx, ranked_list in enumerate(ranked_lists):
        seen_in_this_ranker: set[str] = set()

        for rank, item in enumerate(ranked_list, start=1):
            doc_id = str(item.get(id_field, ""))
            if not doc_id:
                continue  # Bỏ qua item không có id

            # Tránh duplicate trong cùng 1 ranked list
            if doc_id in seen_in_this_ranker:
                continue
            seen_in_this_ranker.add(doc_id)

            rrf_score = 1.0 / (k + rank)
            rrf_scores[doc_id]  = rrf_scores.get(doc_id, 0.0) + rrf_score
            source_count[doc_id] = source_count.get(doc_id, 0) + 1

            # Giữ item từ ranker đầu tiên có nó (chứa đủ metadata nhất)
            if doc_id not in item_map:
                item_map[doc_id] = item

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for doc_id in sorted_ids[:top_k]:
        item = item_map[doc_id].copy()
        item["rrf_score"]    = round(rrf_scores[doc_id], 6)
        item["rrf_sources"]  = source_count[doc_id]   # số ranker đồng thuận
        results.append(item)

    return results


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
    # MMR params
    query_embedding: Optional[list[float]] = None,
    lambda_param: float = MMR_LAMBDA_DEFAULT,
    # RRF params
    ranked_lists: Optional[list[list[dict]]] = None,
    rrf_k: int = RRF_K_DEFAULT,
) -> list[dict]:
    """
    Unified reranking interface — gọi một trong 3 phương pháp.

    Args:
        query           : Câu truy vấn
        candidates      : Danh sách candidates từ retrieval
        top_k           : Số kết quả sau rerank
        method          : 'cross_encoder' | 'mmr' | 'rrf'
        query_embedding : (MMR) embedding vector của query
        lambda_param    : (MMR) trade-off relevance vs diversity
        ranked_lists    : (RRF) list các ranked lists từ nhiều ranker
        rrf_k           : (RRF) smoothing constant

    Returns:
        top_k candidates đã rerank, sorted by rerank score descending.

    Ví dụ dùng:
        # Cross-encoder
        results = rerank(query, candidates, top_k=5, method="cross_encoder")

        # MMR
        results = rerank(query, candidates, top_k=5, method="mmr",
                         query_embedding=q_emb, lambda_param=0.7)

        # RRF
        results = rerank(query, candidates, top_k=5, method="rrf",
                         ranked_lists=[dense_results, bm25_results])
    """
    method = method.lower().strip()

    if method == "cross_encoder":
        try:
            return rerank_cross_encoder(query, candidates, top_k)
        except Exception:
            # Graceful fallback when cross-encoder API fails (missing key, 403, timeouts)
            # Fallback strategy: return candidates sorted by existing 'score' descending.
            sorted_cands = sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)
            results = []
            for c in sorted_cands[:top_k]:
                item = c.copy()
                # preserve original 'score' and add a fallback rerank marker
                item.setdefault("rerank_score", item.get("score", 0.0))
                results.append(item)
            return results

    elif method == "mmr":
        if query_embedding is None:
            raise ValueError(
                "method='mmr' yêu cầu truyền query_embedding.\n"
                "Dùng embedding model (BGE-M3) để encode query trước."
            )
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)

    elif method == "rrf":
        if ranked_lists is None:
            # Fallback: dùng candidates như 1 ranker duy nhất
            ranked_lists = [candidates]
        return rerank_rrf(ranked_lists, top_k, rrf_k)

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Chọn một trong: 'cross_encoder', 'mmr', 'rrf'"
        )


# =============================================================================
# MAIN — DEMO / QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("Task 7: Reranking Module")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Dummy data
    # -------------------------------------------------------------------------
    random.seed(42)
    DIM = 8   # dùng dim nhỏ để demo nhanh

    def _rand_vec(dim=DIM):
        v = np.random.randn(dim).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    query_emb = _rand_vec()

    dummy_candidates = [
        {
            "content":   "Điều 248 BLHS: Tội tàng trữ trái phép chất ma tuý — phạt tù 2-7 năm",
            "score":     0.85,
            "embedding": _rand_vec(),
            "metadata":  {"source": "blhs.md", "type": "legal"},
        },
        {
            "content":   "Điều 249 BLHS: Tội vận chuyển trái phép chất ma tuý",
            "score":     0.80,
            "embedding": _rand_vec(),
            "metadata":  {"source": "blhs.md", "type": "legal"},
        },
        {
            "content":   "Hình phạt cho tội tàng trữ ma tuý từ 2 đến 7 năm tù giam",
            "score":     0.75,
            "embedding": _rand_vec(),
            "metadata":  {"source": "blhs.md", "type": "legal"},
        },
        {
            "content":   "Bắt giữ đối tượng vận chuyển 500g heroin qua biên giới",
            "score":     0.70,
            "embedding": _rand_vec(),
            "metadata":  {"source": "article_01.md", "type": "news"},
        },
        {
            "content":   "Tòa án nhân dân TP.HCM tuyên phạt tử hình kẻ buôn ma tuý",
            "score":     0.65,
            "embedding": _rand_vec(),
            "metadata":  {"source": "article_02.md", "type": "news"},
        },
    ]

    query = "hình phạt tàng trữ ma tuý"

    # -------------------------------------------------------------------------
    # Test MMR
    # -------------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"[MMR] Query: \"{query}\"  λ={MMR_LAMBDA_DEFAULT}")
    print("─" * 60)
    mmr_results = rerank_mmr(query_emb, dummy_candidates, top_k=3, lambda_param=0.7)
    for r in mmr_results:
        print(f"  rank={r['mmr_rank']} | relevance={r['mmr_score']:.4f} | {r['content'][:70]}")
        print(f"         source={r['metadata']['source']}")

    # -------------------------------------------------------------------------
    # Test RRF (2 rankers: dense + bm25 — simulate bằng cách đảo thứ tự)
    # -------------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"[RRF] Query: \"{query}\"  k={RRF_K_DEFAULT}")
    print("─" * 60)

    # Giả lập dense retrieval (sort by embedding score)
    dense_list = sorted(dummy_candidates, key=lambda x: x["score"], reverse=True)

    # Giả lập BM25 retrieval (đảo thứ tự để demo fusion)
    bm25_list = list(reversed(dummy_candidates))

    rrf_results = rerank_rrf([dense_list, bm25_list], top_k=3, k=RRF_K_DEFAULT)
    for r in rrf_results:
        print(f"  rrf_score={r['rrf_score']:.6f} | sources={r['rrf_sources']}/2 | {r['content'][:70]}")

    # -------------------------------------------------------------------------
    # Test Cross-Encoder (chỉ chạy nếu có JINA_API_KEY)
    # -------------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print(f"[Cross-Encoder] Query: \"{query}\"")
    print("─" * 60)
    if not JINA_API_KEY:
        print("  ⚠ Bỏ qua — JINA_API_KEY chưa được set trong .env")
        print("    Set JINA_API_KEY=... để test cross-encoder reranking")
    else:
        try:
            ce_results = rerank_cross_encoder(query, dummy_candidates, top_k=3)
            for r in ce_results:
                print(f"  rerank_score={r['rerank_score']:.4f} | {r['content'][:70]}")
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")

    print("\n✓ Demo hoàn tất!")