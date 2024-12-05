import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import torch

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_chunks.append(text.strip())
    return text_chunks

# Step 2: Split Text into Chunks
def split_into_chunks(text_list, chunk_size=1000, overlap=200):
    chunks = []
    for text in text_list:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
    return chunks

# Step 3: Generate Embeddings for Text Chunks
def generate_embeddings(chunks, model):
    return np.array([model.encode(chunk) for chunk in chunks])

# Step 4: Perform Semantic Search
def semantic_search(query, chunk_embeddings, chunks, embedding_model, top_k=5):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# Step 5: Use a QA Model to Answer Questions
def answer_question(question, context, qa_tokenizer, qa_model):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    outputs = qa_model(**inputs)
    start_scores, end_scores = outputs.start_logits, outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    answer = qa_tokenizer.decode(inputs.input_ids[0][start_index:end_index])
    return answer

# Main Function
def main():
    # Load Local Models
    print("Loading models...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

    # Step 1: Extract Text from PDF
    pdf_path = input("Enter the path to your PDF file: ")
    if not os.path.exists(pdf_path):
        print("❌ PDF file does not exist. Please check the path.")
        return

    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("❌ No extractable text found in the PDF.")
        return

    # Step 2: Split Text into Chunks
    print("Splitting text into chunks...")
    text_chunks = split_into_chunks(pdf_text)

    # Step 3: Generate Embeddings
    print("Generating embeddings...")
    chunk_embeddings = generate_embeddings(text_chunks, embedding_model)

    # Step 4: Chatbot Loop
    print("Ready to answer your questions!")
    while True:
        print("\nOptions:")
        print("1. Ask a question")
        print("2. Exit")
        user_choice = input("Enter your choice (1/2): ")

        if user_choice == "1":
            # Ask a question
            user_question = input("Enter your question: ")
            print("Performing semantic search...")
            top_chunks = semantic_search(user_question, chunk_embeddings, text_chunks, embedding_model)

            # Combine the top chunks for a more comprehensive answer
            print("Answering the question...")
            combined_context = " ".join(top_chunks)
            answer = answer_question(user_question, combined_context, qa_tokenizer, qa_model)
            print(f"\nAnswer: {answer}")

            # Optional: Display full context for verification
            print("\nContext used for the answer:")
            print(combined_context)
        elif user_choice == "2":
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
