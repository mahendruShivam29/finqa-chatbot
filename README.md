# finqa-agentic-rag

Terminal-only FinQA chatbot built as an Agentic-RAG system with LangGraph, hybrid retrieval, and deterministic Python execution for financial calculations.

## Prerequisites

- Python 3.10+
- OpenAI API key with roughly $5 of available credit
- FinQA dataset files: `train.json`, `dev.json`, `test.json`

## Installation

```bash
pip install -r requirements.txt
```

## Data Setup

Clone or download the FinQA dataset from <https://github.com/czyssrs/FinQA> and place these files in `data/`:

- `data/train.json`
- `data/dev.json`
- `data/test.json`

## Environment

Copy `.env.example` to `.env` and fill in your keys:

```env
OPENAI_API_KEY=your_openai_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=finqa-agentic-rag
```

## Run The Chatbot

The chatbot auto-ingests on first run. If `data/faiss_index/` already exists, it loads the persisted index instead of rebuilding it.

```bash
python src/main.py
```

## Run Evaluation

Default 100-sample evaluation:

```bash
python src/eval.py
```

Quick test:

```bash
python src/eval.py --samples 5
```

Optional pooled retrieval stress test:

```bash
python src/eval.py --samples 20 --pooled-eval
```

## Production Serving Note

For GPU production serving, vLLM can replace the OpenAI-hosted reasoning model as a drop-in OpenAI-compatible endpoint. For local CPU inference, Ollama is the practical equivalent. In production, OpenAI embeddings would typically be replaced by TEI or Infinity, and Python execution should move to a sandbox such as E2B or an isolated container.

## Architecture

```text
START --> retrieve --> reason_and_code --> execute
                       ^                 |
                       |  error < 3      |
                       +-----------------+
                                         |
                                         | success OR error >= 3
                                         v
                               generate_final_answer --> END
```
