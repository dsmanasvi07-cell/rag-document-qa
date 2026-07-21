import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        # Load the embedding model once when the object is created
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None          # FAISS index, built once we have chunks
        self.chunks = []           # keep the original text so we can return it later

    def build_index(self, chunks: list[str]):
        """Embed all chunks and build a searchable FAISS index."""
        self.chunks = chunks
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]              # e.g. 384
        self.index = faiss.IndexFlatL2(dimension)    # L2 = Euclidean distance search
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Embed the query and return the top_k most similar chunks."""
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)

        results = [self.chunks[i] for i in indices[0]]
        return results


if __name__ == "__main__":
    from ingest import extract_text_from_pdf, chunk_text

    text = extract_text_from_pdf("data/uploads/sample.pdf")
    chunks = chunk_text(text)

    store = VectorStore()
    store.build_index(chunks)

    query = "What is this document about?"   # change this to something relevant to your PDF
    results = store.search(query, top_k=3)

    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(r[:300], "\n")
        