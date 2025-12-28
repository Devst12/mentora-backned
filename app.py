import pdfplumber
from transformers import pipeline
import torch
import random
import os

class FastAbstractiveSummarizer:
    """Handles the Abstractive Summarization"""
    def __init__(self):
        print("Loading Summarization Model (DistilBART)...")
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",  
            device=-1,
            dtype=torch.float32
        )
        print("Summarization Model Loaded.")

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
        print(f"Processing {len(chunks)} chunks for summary...")

        summaries = []
        with torch.no_grad():
            batch_size = 2
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                result = self.summarizer(
                    batch, max_length=130, min_length=30, do_sample=False, truncation=True
                )
                for r in result:
                    summaries.append(r['summary_text'])
                print(f"Processed chunks {i+1}-{i+len(batch)}")

        return " ".join(summaries)

class MCQGenerator:
    """Handles Generation of Questions and Options"""
    def __init__(self):
        print("Loading Question Generation Model (T5-Small)...")
        # T5 is excellent at generating questions from text
        self.qg_pipeline = pipeline(
            "text2text-generation",
            model="t5-small",
            device=-1,
            torch_dtype=torch.float32
        )
        print("Question Model Loaded.")

    def _extract_context_words(self, text):
        """Extracts candidate words for answer options (distractors)"""
        # Simple approach: Get long words (nouns usually)
        words = []
        tokens = text.split()
        for t in tokens:
            # Keep words longer than 5 chars as candidates
            if len(t) > 5 and t.isalpha():
                words.append(t)
        return list(set(words)) # Remove duplicates

    def generate_mcqs(self, summary_text, num_questions=10):
        """
        Generates Questions + Answers using AI.
        Generates Wrong Options using random words from text.
        """
        print(f"\nGenerating {num_questions} Questions... (This requires thinking)")
        
        # We split summary into parts to generate questions
        sentences = summary_text.split('. ')
        questions_list = []
        
        # We will attempt to generate 1 question per sentence until we reach the count
        for i, sentence in enumerate(sentences[:num_questions+5]):
            if len(questions_list) >= num_questions:
                break
            
            # Prepare input for T5
            # T5 expects a prefix instruction
            input_text = f"generate question: {sentence}"
            
            # Generate
            result = self.qg_pipeline(
                input_text, 
                max_length=64, 
                num_return_sequences=1, 
                do_sample=False
            )
            
            question = result[0]['generated_text']
            
            # Heuristic: Try to extract the answer (last word or noun)
            # T5 often outputs "question: ... answer: ..." or just the question.
            # We will treat the last word of the generated question as a potential answer guess
            # or just create a placeholder if it's just a question.
            
            # For simplicity in this script, we will assume the AI generates a question.
            # We'll pick the answer from the source sentence logic manually if T5 doesn't separate it.
            # However, T5 usually just outputs the question string.
            
            # Let's pick a random noun from the sentence as the "Correct Answer"
            candidates = [w for w in sentence.split() if len(w) > 4]
            if not candidates:
                continue
            
            answer = candidates[0] # Naively take the first long word as answer
            
            # Get Distractors (Wrong answers)
            context_words = self._extract_context_words(summary_text)
            # Remove the correct answer from context words to avoid duplicates
            if answer in context_words:
                context_words.remove(answer)
            
            # Pick 3 random wrong answers
            if len(context_words) < 3:
                wrong_answers = ["Concept A", "Concept B", "Concept C"] # Fallback
            else:
                wrong_answers = random.sample(context_words, 3)
            
            # Combine and Shuffle
            options = wrong_answers + [answer]
            random.shuffle(options)
            
            # Find index of correct answer
            correct_index = options.index(answer)
            correct_letter = chr(65 + correct_index) # 0 -> A, 1 -> B
            
            questions_list.append({
                "question": question,
                "options": options,
                "correct": correct_letter,
                "correct_text": answer
            })

        return questions_list

# ================= EXECUTION =================

def main():
    pdf_path = input("Enter PDF path: ").strip()
    
    # 1. Initialize Summarizer
    summarizer = FastAbstractiveSummarizer()

    print("\n[1/3] Extracting text...")
    text = summarizer.extract_text(pdf_path)
    if not text:
        print("No text found.")
        return

    print("[2/3] Generating abstractive summary...")
    final_summary = summarizer.generate_summary(text)

    print("\n" + "="*70)
    print("ABSTRACTIVE SUMMARY")
    print("="*70)
    print(final_summary)
    print("="*70)

    # 2. MCQ Interaction
    choice = input("\nDo you want to generate 10 MCQs based on this summary? (yes/no): ").strip().lower()
    
    if choice in ['yes', 'y']:
        # Initialize Generator only if needed
        mcq_gen = MCQGenerator()
        
        print("\n[3/3] Generating MCQs...")
        mcqs = mcq_gen.generate_mcqs(final_summary, num_questions=10)
        
        print("\n" + "="*70)
        print("GENERATED MCQS")
        print("="*70)
        
        for i, q in enumerate(mcqs, 1):
            print(f"\nQ{i}: {q['question']}")
            print(f"   A) {q['options'][0]}")
            print(f"   B) {q['options'][1]}")
            print(f"   C) {q['options'][2]}")
            print(f"   D) {q['options'][3]}")
            print(f"   [Correct Answer: {q['correct']}: {q['correct_text']}]")
        
        print("="*70)
    else:
        print("Process finished.")

if __name__ == "__main__":
    main()