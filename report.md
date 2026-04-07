# FinQA Agentic-RAG Report

## Introduction

FinQA is a financial question-answering benchmark built from earnings reports. Each sample mixes unstructured paragraphs with structured tables, and the answer usually depends on multi-step numerical reasoning across both modalities. That makes the task materially harder than standard QA: extracting the right facts is not enough, because the system must also compute the exact result.

## Methodology

The system uses an Agentic-RAG architecture with four LangGraph nodes: `retrieve`, `reason_and_code`, `execute`, and `generate_final_answer`. Text is chunked strictly at the paragraph level so `gold_inds` keys such as `text_3` stay aligned with retrieval units. If a single paragraph exceeds 512 tokens, it is split with `RecursiveCharacterTextSplitter` while preserving the original `chunk_index`. Tables are never split; each table is serialized once to Markdown and kept intact as `table_0`.

For global interactive retrieval, the system uses a multi-vector design for tables. A GPT-4o-mini summary of each table is embedded into FAISS, while the original raw Markdown table is stored as the parent document. Retrieval returns the raw table, not the summary. BM25 indexes text chunks plus both the raw table Markdown and the table summary. Hybrid retrieval is implemented as a two-stage merge because `EnsembleRetriever` does not cleanly combine BM25 summary hits with `MultiVectorRetriever` parent hits. BM25 table hits are resolved back to the raw table parent using `parent_id`, deduplicated so only the highest-ranked BM25 hit per parent survives, and then merged with FAISS parent hits using reciprocal rank fusion with BM25 weight `0.4` and FAISS weight `0.6`.

The execution path is intentionally deterministic. The model writes Python, and the Python REPL computes the answer. This avoids arithmetic hallucination that would otherwise make standard RAG unreliable on FinQA. The current implementation uses `PythonREPLTool` locally, but this must be replaced with a sandbox such as E2B or a containerized executor in production.

Evaluation uses a different retrieval scope by design. Each dev-set sample builds its own temporary in-memory retriever over only that sample's `pre_text`, `post_text`, and `table`. This mirrors the FinQA benchmark protocol and isolates reasoning quality from cross-report search noise. The limitation is explicit: retrieval precision is easier in this setting because each index contains only a handful of documents. To offset that, the evaluation script also supports an optional pooled hard mode that indexes 10-sample clusters together.

## Cost Management

The system follows the SDD cost plan closely:

- `gpt-4o-mini` for reasoning, code generation, final answer generation, and table summarization
- `gpt-4o` only for the 10-sample LLM-as-a-judge pass
- `text-embedding-3-small` for embeddings

This keeps ingestion and evaluation within the stated sub-$5 budget while reserving margin for retries and debugging.

## Evaluation Results

Run `python src/eval.py` after adding dataset files and API credentials. The script reports:

- Exact Match Accuracy with `1e-4` tolerance
- Retrieval Precision and Recall as macro-averages
- LLM-as-a-Judge reasoning score on the first 10 samples
- Average latency per question

Fill the table below after running evaluation:

| Metric | Value |
|---|---|
| Exact Match Accuracy | TBD |
| Retrieval Precision | TBD |
| Retrieval Recall | TBD |
| Average Latency | TBD |
| Reasoning Quality Score | TBD |

## Production Considerations

LangSmith tracing is enabled through `.env` with `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_PROJECT=finqa-agentic-rag`. This supports node-level tracing, token accounting, latency inspection, and tool error monitoring. For production deployment, a GPU-hosted vLLM server can replace the OpenAI reasoning endpoint with no graph-level changes, while Ollama is the CPU-oriented local alternative. Embeddings should move to TEI or Infinity for self-hosted serving. Code execution should move from `PythonREPLTool` to E2B or an equivalent sandbox, and drift should be monitored through tool error rate plus periodic exact-match checks on a holdout set.
