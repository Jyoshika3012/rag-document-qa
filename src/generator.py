import groq
import os
from retriever import retrieve_chunks
from dotenv import load_dotenv

load_dotenv()

client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(query: str, index_path: str = "data/faiss_index"):
    docs = retrieve_chunks(query, index_path)

    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', '?')+1}]: {doc.page_content}"
        for doc in docs
    ])

    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't have enough information to answer that."
Always mention which part of the document your answer comes from.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    question = "What is this document about?"
    print(f"Question: {question}\n")
    answer = answer_question(question)
    print(f"Answer: {answer}")