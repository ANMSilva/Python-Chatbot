from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch

# Check system setup
print("Torch CUDA available:", torch.cuda.is_available())

# Load a PDF
pdf_path = "test.pdf"  # Replace with the path to your PDF
try:
    reader = PdfReader(pdf_path)
    text = reader.pages[0].extract_text()  # Extract text from the first page
    print(f"Extracted text from the first page: {text[:100]}...\n")  # Preview first 200 chars

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([text])  # Get embeddings for the text
    print(f"Text embedding shape: {embeddings.shape}")
except Exception as e:
    print(f"Error: {e}")
