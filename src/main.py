from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from graph import build_graph
from ingest import load_or_build_index
from tools import configure_retriever_tool


def main() -> None:
    load_dotenv()
    retriever = load_or_build_index()
    configure_retriever_tool(retriever)
    compiled_graph = build_graph()

    while True:
        question = input("\nEnter a financial question (or 'quit' to exit): ").strip()
        if question.lower() == "quit":
            break

        print("\n[1/4] Retrieving relevant financial documents...")
        for step in compiled_graph.stream(
            {"messages": [HumanMessage(content=question)], "error_count": 0}
        ):
            node_name = list(step.keys())[0]
            node_output = step[node_name]

            if node_name == "retrieve":
                ctx = node_output.get("retrieved_context", "")
                print(f"[2/4] Retrieved context ({len(ctx)} chars)")
            elif node_name == "reason_and_code":
                print(f"[3/4] Generated code:\n{node_output.get('generated_code', '')}")
            elif node_name == "execute":
                print(f"      Execution result: {node_output.get('execution_result', '')}")
            elif node_name == "generate_final_answer":
                print(f"\n[4/4] Answer: {node_output.get('final_answer', '')}")


if __name__ == "__main__":
    main()
