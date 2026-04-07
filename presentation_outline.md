# Presentation Outline

| Slide | Content |
|---|---|
| 1 | **Title & Problem Statement** - why financial QA is uniquely hard: numerical hallucination risk, exact arithmetic requirements, and text-plus-table reasoning |
| 2 | **Data Understanding** - FinQA sample structure, paragraph-level chunking, why merge-then-split breaks `gold_inds`, per-sample scoped retrieval and its limitations |
| 3 | **Architecture** - LangGraph four-node flow, retry loop, empty-code guard, GPT-4o-mini as the primary model and GPT-4o as judge-only |
| 4 | **Retrieval Deep-Dive** - summary-to-FAISS plus UUID-to-raw-table mapping, BM25 over text plus raw tables plus summaries, RRF merge, parent dedup logic |
| 5 | **Demo** - terminal walkthrough with streamed retrieval, code generation, execution, and final answer output |
| 6 | **Evaluation** - EM, macro-avg precision and recall, LLM-as-a-judge score, single-step percentage normalization, LangSmith trace screenshot |
| 7 | **Path to Production** - vLLM on GPU, Ollama on CPU, TEI for embeddings, E2B sandboxing, and LangSmith-based drift monitoring |
