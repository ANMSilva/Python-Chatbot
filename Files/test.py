import PyPDF2
from sentence_transformers import SentenceTransformer
import torch

print("PyPDF2 version:", PyPDF2.__version__)
print("Sentence Transformers version:", SentenceTransformer._version)
print("Torch version:", torch.__version__)