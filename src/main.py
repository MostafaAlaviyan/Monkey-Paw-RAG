from retrieve import Retriever
from chat import generate_answer


def main():

    print("=" * 60)
    print("        The Monkey's Paw - RAG Chatbot")
    print("=" * 60)

    retriever = Retriever()

    print("\nRAG chatbot is ready.")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nSearching the story...")

        context = retriever.retrieve(
            question,
            top_k=3
        )

        print("Generating answer...\n")

        answer = generate_answer(
            question,
            context
        )

        print("Assistant:")
        print(answer)

        print()


if __name__ == "__main__":
    main()