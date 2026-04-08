import argparse
import json
import re
import time
from statistics import mean

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from graph import build_graph
from ingest import (
    DATA_DIR,
    build_eval_sample_retriever,
    build_pooled_eval_retriever,
    get_embedding_model,
    load_json_file,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--pooled-eval",
        action="store_true",
        help="Run additional pooled retrieval eval",
    )
    return parser.parse_args()


def parse_ground_truth(sample: dict) -> float:
    raw = sample["qa"]["exe_ans"]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.endswith("%"):
            return float(raw.rstrip("%")) / 100.0
        return float(raw)
    return float(raw)


def extract_predicted_answer(text: str) -> float | None:
    matches = re.findall(r"([-+]?\d+\.?\d*|[-+]?\.\d+)\s*(%)?", text)
    if not matches:
        return None
    last_num, pct_sign = matches[-1]
    value = float(last_num)
    if pct_sign == "%":
        value /= 100.0
    return value


def check_answer(predicted: float, ground_truth: float, tol: float = 1e-4) -> bool:
    if abs(predicted - ground_truth) <= tol:
        return True
    if abs(predicted * 100 - ground_truth) <= tol:
        return True
    if abs(predicted / 100 - ground_truth) <= tol:
        return True
    return False


def judge_reasoning(
    judge_llm: ChatOpenAI,
    question: str,
    ground_truth: float,
    generated_code: str,
    final_answer_text: str,
) -> bool:
    response = judge_llm.invoke(
        (
            f"Question: {question}\n"
            f"Ground Truth Answer: {ground_truth}\n"
            f"Agent Reasoning and Code: {generated_code}\n"
            f"Agent Final Answer: {final_answer_text}\n\n"
            "Did the agent retrieve the correct data and reason correctly to arrive at "
            "its answer? Reply with Yes or No followed by one sentence of justification."
        )
    )
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content.strip().lower().startswith("yes")


def pooled_metrics(samples: list[dict], embeddings) -> tuple[float, float]:
    pooled_precisions: list[float] = []
    pooled_recalls: list[float] = []

    for start in range(0, len(samples), 10):
        cluster = samples[start : start + 10]
        if not cluster:
            continue
        retriever = build_pooled_eval_retriever(cluster, embeddings=embeddings)
        for sample in cluster:
            docs = retriever.invoke(sample["qa"]["question"], k=3)
            gold_keys = set(sample["qa"]["gold_inds"].keys())
            retrieved_keys = {
                f"{doc.metadata['type']}_{doc.metadata['chunk_index']}" for doc in docs
            }
            hits = len(gold_keys & retrieved_keys)
            pooled_precisions.append(hits / len(retrieved_keys) if retrieved_keys else 0.0)
            pooled_recalls.append(hits / len(gold_keys) if gold_keys else 0.0)

    return (
        mean(pooled_precisions) if pooled_precisions else 0.0,
        mean(pooled_recalls) if pooled_recalls else 0.0,
    )


def main() -> None:
    args = parse_args()
    samples = load_json_file(DATA_DIR / "dev.json")[: args.samples]
    embeddings = get_embedding_model()
    judge_llm = ChatOpenAI(model="gpt-4o")
    results_path = DATA_DIR / "eval_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")

    correct_count = 0
    latencies: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    judge_scores: list[bool] = []

    for idx, sample in enumerate(samples):
        question = sample["qa"]["question"]
        ground_truth = parse_ground_truth(sample)
        retriever = build_eval_sample_retriever(sample, embeddings=embeddings)
        retrieved_docs = retriever.invoke(question, k=3)

        gold_keys = set(sample["qa"]["gold_inds"].keys())
        retrieved_keys = {
            f"{doc.metadata['type']}_{doc.metadata['chunk_index']}" for doc in retrieved_docs
        }
        hits = len(gold_keys & retrieved_keys)
        precision_i = hits / len(retrieved_keys) if retrieved_keys else 0.0
        recall_i = hits / len(gold_keys) if gold_keys else 0.0
        precisions.append(precision_i)
        recalls.append(recall_i)

        graph = build_graph(scoped_retriever=retriever)
        start_time = time.time()
        result = graph.invoke({"messages": [HumanMessage(content=question)], "error_count": 0})
        latency = time.time() - start_time
        latencies.append(latency)

        final_answer_text = result["final_answer"]
        predicted = extract_predicted_answer(final_answer_text)
        correct = predicted is not None and check_answer(predicted, ground_truth)
        correct_count += int(correct)

        if idx < 10:
            judge_scores.append(
                judge_reasoning(
                    judge_llm,
                    question,
                    ground_truth,
                    result.get("generated_code", ""),
                    final_answer_text,
                )
            )

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sample_id": sample["id"],
                        "question": question,
                        "ground_truth": ground_truth,
                        "predicted": predicted,
                        "correct": correct,
                        "final_answer": final_answer_text,
                        "generated_code": result.get("generated_code", ""),
                        "error_count": result.get("error_count", 0),
                        "latency_s": latency,
                        "precision": precision_i,
                        "recall": recall_i,
                    }
                )
                + "\n"
            )

    em_accuracy = correct_count / len(samples) if samples else 0.0
    retrieval_precision = mean(precisions) if precisions else 0.0
    retrieval_recall = mean(recalls) if recalls else 0.0
    average_latency = mean(latencies) if latencies else 0.0
    reasoning_score = sum(judge_scores)

    print(f"=== FinQA Evaluation Results ({len(samples)} samples) ===")
    print(f"Exact Match Accuracy (+/-1e-4):     {em_accuracy * 100:.1f}%")
    print(f"Retrieval Precision (macro-avg):  {retrieval_precision * 100:.1f}%")
    print(f"Retrieval Recall (macro-avg):     {retrieval_recall * 100:.1f}%")
    print(f"Reasoning Quality Score (LLM-as-a-Judge, n=10): {reasoning_score}/10")
    print(f"Average Latency per Question:     {average_latency:.2f}s")
    print(f"Detailed results saved to:        {results_path}")

    if args.pooled_eval:
        pooled_precision, pooled_recall = pooled_metrics(samples, embeddings)
        print(f"Pooled_Precision:                 {pooled_precision * 100:.1f}%")
        print(f"Pooled_Recall:                    {pooled_recall * 100:.1f}%")


if __name__ == "__main__":
    main()
