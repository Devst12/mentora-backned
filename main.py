import pdfplumber
from transformers import pipeline
import torch
import os

class FastAbstractiveSummarizer:
    def __init__(self):
        print("Loading CPU-optimized abstractive model...")
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",  
            device=-1,  # CPU
            dtype=torch.float32
        )
        print("Model Loaded.")

    def extract_text(self, pdf_path):
        if not os.path.exists(pdf_path):
            return ""
        text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            return "\n".join(text)
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def chunk_text(self, text, max_chars=2000):
        sentences = text.split('. ')
        chunks, chunk = [], ""
        for s in sentences:
            if len(chunk) + len(s) < max_chars:
                chunk += s + ". "
            else:
                chunks.append(chunk.strip())
                chunk = s + ". "
        if chunk:
            chunks.append(chunk.strip())
        return chunks

    def generate_summary(self, text):
        if not text.strip():
            return "No text to summarize."

        chunks = self.chunk_text(text)
        print(f"Processing {len(chunks)} chunks...")

        summaries = []
        with torch.no_grad():  # CPU-friendly inference
            batch_size = 2  # Summarize 2 chunks at once
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                result = self.summarizer(
                    batch,
                    max_length=130,
                    min_length=30,
                    do_sample=False,
                    truncation=True
                )
                for r in result:
                    summaries.append(r['summary_text'])
                print(f"Processed chunks {i+1}-{i+len(batch)}")

        return " ".join(summaries)

# ================= EXECUTION =================

def main():
    pdf_path = input("Enter PDF path: ").strip()
    summarizer = FastAbstractiveSummarizer()

    print("Extracting text...")
    text = summarizer.extract_text(pdf_path)
    if not text:
        print("No text found in PDF.")
        return

    print("Generating abstractive summary...")
    final_summary = summarizer.generate_summary(text)

    print("\n" + "="*70)
    print("ABSTRACTIVE SUMMARY")
    print("="*70)
    print(final_summary)
    print("="*70)

if __name__ == "__main__":
    main()
