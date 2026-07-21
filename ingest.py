from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    """Reads a PDF and returns all its text as one big string."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into overlapping chunks of roughly `chunk_size` words.
    Overlap ensures we don't lose context that falls right at a chunk boundary.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward, but re-include the last `overlap` words

    return chunks


if __name__ == "__main__":
    # Quick test — put any PDF in data/uploads and update the filename below
    text = extract_text_from_pdf("data/uploads/sample.pdf")
    chunks = chunk_text(text)
    print(f"Extracted {len(text)} characters")
    print(f"Split into {len(chunks)} chunks")
    print("First chunk preview:\n", chunks[0][:300])
    
