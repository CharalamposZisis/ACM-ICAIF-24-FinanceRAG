from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from light_embed import TextEmbedding
import numpy as np
from load_data import load_data
from preprocess_data import TextPreprocessor
from langchain.schema import Document

class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = TextEmbedding(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")
    
    def dict_to_documents(self, dict_docs: List[dict]) -> List[Document]:
        
        documents = []
        for doc in dict_docs:
            doc_obj = Document(
                page_content=doc.get('text', ''),
                metadata={
                    '_id': doc.get('_id', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', 'unknown')
                }
            )
            documents.append(doc_obj)
        return documents

    def preprocess_documents(self, documents: List[Document], 
                           preprocessor: TextPreprocessor) -> List[Document]:
        """Κάνε preprocessing στα documents"""
        processed = []
        for doc in documents:
            processed_content = preprocessor.process(doc.page_content)
            processed_doc = Document(
                page_content=processed_content,
                metadata=doc.metadata
            )
            processed.append(processed_doc)
        return processed
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        json_chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(json_chunks)} chunks.")
        return json_chunks
    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
    
if __name__ == "__main__":
    
    docs = load_data()
    preprocessor = TextPreprocessor()

    processed_docs = []

    for doc in docs:
        processed_doc = doc.copy()

        if "text" in processed_doc:
            processed_doc["text"] = preprocessor.process(processed_doc["text"])

        processed_docs.append(processed_doc)
        
    emb_pipe = EmbeddingPipeline()
    
    documents = emb_pipe.dict_to_documents(processed_docs)
    
    chunks = emb_pipe.chunk_documents(documents)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)