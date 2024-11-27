# test_client.py
from langserve import RemoteRunnable
import json
from typing import Optional, Dict, List
import time

class RAGTester:
    def __init__(self, base_url: str = "http://0.0.0.0:9012"):
        self.base_url = base_url
        self.retriever = RemoteRunnable(f"{base_url}/retriever/")
        self.generator = RemoteRunnable(f"{base_url}/generator/")

    def test_retriever(self, query: str) -> List[Dict]:
        """Test retriever endpoint"""
        print(f"\nTesting Retriever with query: '{query}'")
        try:
            result = self.retriever.invoke({"input": query})
            print("\nRetriever Results:")
            print(json.dumps(result, indent=2))
            return result
        except Exception as e:
            print(f"Retriever Error: {e}")
            return []

    def test_generator(self, query: str, context: Optional[List[Dict]] = None) -> str:
        """Test generator endpoint"""
        try:
            input_data = {
                "input": query,
                "context": context if context else []
            }
            result = self.generator.invoke(input_data)
            print("\nGenerator Results:")
            print(result)
            return result
        except Exception as e:
            print(f"Generator Error: {e}")
            return str(e)

    def test_rag_pipeline(self, query: str):
        """Test complete RAG pipeline"""
        print(f"\n{'='*50}")
        print(f"Testing RAG Pipeline with query: '{query}'")
        
        # Get relevant documents
        start_time = time.time()
        docs = self.test_retriever(query)
        retriever_time = time.time() - start_time
        
        # Generate response
        start_time = time.time()
        response = self.test_generator(query, docs)
        generator_time = time.time() - start_time
        
        print(f"\nTiming:")
        print(f"Retriever: {retriever_time:.2f}s")
        print(f"Generator: {generator_time:.2f}s")
        print(f"Total: {retriever_time + generator_time:.2f}s")
        print('='*50)
        
        return response

def main():
    # Initialize tester
    tester = RAGTester()
    
    # Test queries
    test_queries = [
        "Tell me about attention is all you need",
        "What is spatialVLM?",
        "Explain the transformer architecture",
        "What are the key components of deep learning?",
    ]
    
    # Run tests
    for query in test_queries:
        tester.test_rag_pipeline(query)

if __name__ == "__main__":
    main()
