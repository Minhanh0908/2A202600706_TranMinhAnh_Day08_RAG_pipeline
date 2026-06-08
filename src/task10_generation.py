"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if not chunks:
        return []
    if len(chunks) <= 2:
        return chunks

    # Take every other chunk starting from best (index 0): 0,2,4,...
    first_pass = [chunks[i] for i in range(0, len(chunks), 2)]
    # Then append the remaining ones in reverse order: (last odd, ..., 1)
    second_pass = [chunks[i] for i in range(len(chunks) - 1 if (len(chunks) - 1) % 2 == 1 else len(chunks) - 2, 0, -2)]
    # The above range computes the largest odd index ≤ len(chunks)-1
    reordered = first_pass + second_pass
    # Ensure stable length
    return reordered[: len(chunks)]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source") or meta.get("source_path") or f"document-{i}"
        # include filename without extension for concise citation label
        src_label = os.path.splitext(str(source))[0]
        doc_type = meta.get("type", "unknown")
        header = f"[Document {i} | Source: {src_label} | Type: {doc_type}]"
        body = chunk.get("content", "")
        parts.append(f"{header}\n{body}")
    return "\n---\n".join(parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = []
    try:
        chunks = retrieve(query, top_k=top_k) or []
    except Exception:
        chunks = []

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    # Step 5: Call LLM if available; otherwise fallback to extractive answer
    answer_text = ""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=os.getenv("PAGEINDEX_RETRIEVE_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=float(TEMPERATURE),
                top_p=float(TOP_P),
            )
            # openai.ChatCompletions format
            answer_text = response.choices[0].message.content
        except Exception:
            answer_text = ""

    if not answer_text:
        # Fallback: extractive summary from top chunks with citation markers
        if not reordered:
            answer_text = "Tôi không thể xác minh thông tin này từ nguồn hiện có"
        else:
            parts = []
            for i, c in enumerate(reordered[: top_k], 1):
                src = c.get("metadata", {}).get("source", "unknown")
                src_label = os.path.splitext(str(src))[0]
                snippet = c.get("content", "").strip().replace("\n", " ")[:300]
                parts.append(f"{snippet} [{src_label}]")
            answer_text = "\n\n".join(parts)

    return {
        "answer": answer_text,
        "sources": reordered,
        "retrieval_source": reordered[0].get("source", "hybrid") if reordered else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
