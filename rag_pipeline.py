import os
from dotenv import load_dotenv
from groq import Groq
from vector_store import VectorStore
from ingest import extract_text_from_pdf, chunk_text

load_dotenv()  # reads .env and loads GROQ_API_KEY into environment

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Send the question + retrieved context to the LLM and get an answer."""
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context provided below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    text = extract_text_from_pdf("data/uploads/sample.pdf")
    chunks = chunk_text(text)

    store = VectorStore()
    store.build_index(chunks)

    question = "What is the capital of france?"   # change to something relevant to your PDF
    retrieved_chunks = store.search(question, top_k=3)

    answer = generate_answer(question, retrieved_chunks)

    print(f"Question: {question}\n")
    print(f"Answer: {answer}")