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

## Architecture Diagrams

![Data Ingestion And Persistence](data_ingestion_and_persistence.png)

![Inference And Agentic Loop](inference_and_agentic_loop.png)

![Evaluation And Monitoring](evaluation_and_monitoring.png)

## The "Why" Behind the Architecture

### 1. The "LLMs Can't Do Math" Problem

The primary motivation is to move from **Probabilistic Math** to **Deterministic Math**.

**The Problem:** LLMs are "Next Token Predictors." When you ask an LLM to calculate a CAGR (Compound Annual Growth Rate), it isn't using a calculator; it is guessing what the next number should look like based on its training data. In finance, being "close enough" is a failure.

**The Solution:** This project uses the LLM only for logic (writing the formula) and offloads the calculation to a Python interpreter. This ensures that if the logic is right, the math is 100% perfect.

---

### 2. The "Table Retrieval" Challenge

Most RAG (Retrieval-Augmented Generation) systems are built for text (like Wikipedia articles). They struggle with financial reports because the "meat" of the data is trapped in tables.

**The Problem:** If you turn a table into a vector (a string of numbers), the semantic meaning often gets lost. A search for "2022 Revenue" might fail because the number is just one cell in a giant grid.

**The Solution:** This project uses **Table Summarization** (what you are doing now) and **Hybrid Search**. By having an LLM describe the table in English first, we create a "searchable bridge" that allows the system to find the right data even when the user's question is vague.

---

### 3. Moving from "Chains" to "Agents" (Self-Correction)

Standard AI pipelines are "Linear Chains": Search â†’ Context â†’ Answer. If the search finds the wrong data, the answer is wrong, and the process ends.

**The Problem:** Financial questions are often multi-step. You might need to find one value, then find another, then compare them. A linear chain often trips up on these.

**The Solution:** By using **LangGraph**, the project builds an **Agent**. If the Python code fails because a variable is missing, the agent "realizes" it made a mistake, goes back to the documents, retrieves the missing number, and tries again. This "Self-Correction" loop mimics how a human analyst actually works.

---

### 4. Cost-Effective Intelligence

**The Problem:** Using massive models like `gpt-4o` for every single task is expensive and slow.

**The Solution:** This project is motivated by the idea of **"Agentic Reasoning over Model Size."** By using a smaller, cheaper model (`gpt-4o-mini`) but giving it a structured workflow, a Python tool, and a self-correction loop, we can achieve accuracy levels that rival (or beat) a much larger model used in a simple chat interface.

---

### 5. The "FinQA" Benchmark

Finally, the project is motivated by the **FinQA Dataset** itself. It is one of the most difficult benchmarks in AI because it requires:

- **Deep Retrieval:** Finding the right page in a 10-K filing.
- **Numerical Reasoning:** Understanding "millions" vs "billions" and "increase" vs "decrease."
- **Program Synthesis:** Turning a financial question into a multi-step math program.
