from ollama import chat

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "qwen2.5:0.5b" # "llama3.1:8b"

# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are a question-answering assistant for the story
"The Monkey's Paw".

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say:

"I cannot find the answer in the provided story."

Keep your answer clear and concise.
"""

# ---------------------------------------------------------
# Generate answer
# ---------------------------------------------------------

def generate_answer(
    question: str,
    context: list[str]
) -> str:

    context_text = "\n\n".join(
        context
    )

    user_prompt = f"""
Context from The Monkey's Paw:

-------------------------
{context_text}
-------------------------

Question:

{question}

Answer based only on the context above.
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    return response.message.content
