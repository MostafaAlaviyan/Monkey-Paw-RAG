from retrieve import Retriever
from chat import generate_answer


def main():
    retriever = Retriever()

    print("Monkey's Paw RAG")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ")

        if question.lower() == "exit":
            break

        # Retrieve relevant chunks
        results = retriever.retrieve(
            question
        )

        # Extract retrieved documents for the LLM
        context = results["documents"]
        

        # Generate answer
        answer = generate_answer(
            question,
            context
        )

        print("\nAnswer:")
        print(answer)

        # Optional: show retrieved chunks
        print("\nRetrieved chunks:")
        for rank, chunk_id in enumerate(
            results["ids"],
            start=1
        ):
            print(f"{rank}. {chunk_id}")

        print()


if __name__ == "__main__":
    main()