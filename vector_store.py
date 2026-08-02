import os
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "models/text-embedding-001"


def get_embedding(text: str) -> list[float]:
    result = genai.embed_content(model=EMBED_MODEL, content=text)
    return result["embedding"]


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[str]):
        self.chunks = chunks
        embeddings = [get_embedding(chunk) for chunk in chunks]
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        query_vector = np.array([get_embedding(query)]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        return [self.chunks[i] for i in indices[0]]