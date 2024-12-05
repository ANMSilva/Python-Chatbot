from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

def extract_pdf_text(pdf_path):
    """
    Extracts text from a PDF and organizes it by page.
    """
    reader = PdfReader(pdf_path)
    text_data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_data.append({"page": i + 1, "text": text.strip()})
    return text_data

def generate_embeddings(text_data, model):
    """
    Generates embeddings for the extracted text using a pre-trained model.
    """
    embeddings = []
    for entry in text_data:
        text_embedding = model.encode(entry["text"], convert_to_tensor=True)
        embeddings.append({"page": entry["page"], "text": entry["text"], "embedding": text_embedding})
    return embeddings

def chatbot():
    print("📄 Welcome to the Vector-based PDF Chatbot!")
    print("You can upload a PDF and interact with its content.")
    
    # Step 1: Ask for the PDF file
    pdf_path = input("Please enter the path to your PDF file: ")
    
    try:
        # Load the PDF and extract text
        print("⏳ Extracting text from the PDF...")
        text_data = extract_pdf_text(pdf_path)
        
        if not text_data:
            print("❌ No extractable text found in the PDF.")
            return
        
        print(f"✅ Extracted text from {len(text_data)} pages.")
        
        # Load pre-trained model for embeddings
        print("⏳ Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = generate_embeddings(text_data, model)
        print("✅ Embeddings generated.")
        
        # Chat loop
        while True:
            print("\nOptions:")
            print("1. Ask a question")
            print("2. Exit chatbot")
            
            user_choice = input("Please enter your choice (1/2): ")
            
            if user_choice == "1":
                # Ask a question
                user_query = input("What do you want to know? ")
                query_embedding = model.encode(user_query, convert_to_tensor=True)
                
                # Find the most relevant text based on semantic similarity
                highest_score = -1
                best_match = None
                for entry in embeddings:
                    similarity = util.pytorch_cos_sim(query_embedding, entry["embedding"]).item()
                    if similarity > highest_score:
                        highest_score = similarity
                        best_match = entry
                
                # Display the result with improved context
                if best_match:
                    print(f"\n📄 Best Match from Page {best_match['page']} (Score: {highest_score:.2f}):")
                    print(best_match["text"])
                else:
                    print("❌ No relevant content found.")
            
            elif user_choice == "2":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
    
    except FileNotFoundError:
        print("❌ File not found. Please check the file path and try again.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()

#new commit