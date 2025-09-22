import os
import re
import markdown
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple

class RAGSystem:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
        self.index = None
        
    def load_markdown_document(self, file_path: str):
        """Load and process a markdown document for RAG"""
        if not os.path.exists(file_path):
            print(f"Document not found: {file_path}")
            return False
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to plain text
        html = markdown.markdown(content)
        text = re.sub('<[^<]+?>', '', html)
        
        # Split into chunks (paragraphs)
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 50]
        
        self.documents = chunks
        print(f"Loaded {len(chunks)} document chunks")
        
        # Create embeddings
        self._create_embeddings()
        return True
    
    def _create_embeddings(self):
        """Create FAISS index from document chunks"""
        if not self.documents:
            return
            
        print("Creating embeddings...")
        embeddings = self.embedder.encode(self.documents)
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Created FAISS index with {len(self.documents)} documents")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        if not self.index or not self.documents:
            return ""
        
        # Encode query
        query_embedding = self.embedder.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get relevant chunks
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if scores[0][i] > 0.3:  # Similarity threshold
                relevant_chunks.append(self.documents[idx])
        
        return "\n\n".join(relevant_chunks)
    
    def is_loaded(self) -> bool:
        """Check if RAG system has documents loaded"""
        return len(self.documents) > 0