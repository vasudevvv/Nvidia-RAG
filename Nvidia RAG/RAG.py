# server_app.py
from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Union
from langchain_core.documents import Document
import os

# Initialize components
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama3-8b-instruct")

# Load the local FAISS index
if os.path.exists("docstore_index"):
    vector_store = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
else:
    # Fallback sample docs for testing if index doesn't exist
    sample_texts = [
        "SpatialVLM is a vision-language model that specializes in spatial reasoning tasks.",
        "The model can understand and reason about spatial relationships in images.",
        "It uses attention mechanisms to process visual and textual information.",
        "The architecture is based on the transformer model with spatial awareness components."
    ]
    vector_store = FAISS.from_texts(sample_texts, embedder)

def extract_query(data: Union[str, Dict]) -> str:
    """Extract query from various input formats"""
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return data.get("input", "")
    return str(data)

class Retriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def __call__(self, query_or_data: Union[str, Dict]) -> List[Dict]:
        query = extract_query(query_or_data)
        try:
            docs = self.vectorstore.similarity_search(query, k=4)
            return [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"Retriever error: {e}")
            return []

# Create retriever chain
retriever = Retriever(vector_store)
retriever_chain = RunnableLambda(retriever)

def format_docs(docs: List[Dict]) -> str:
    """Format documents into a single string"""
    return "\n".join(doc.get("page_content", "") for doc in docs)

# Generator chain
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, acknowledge this and provide general information while being clear about the distinction."),
    ("user", """Context: {context}

Question: {input}

Please provide a detailed answer based on the context above.""")
])

def process_generator_input(data: Dict) -> Dict:
    """Process input for generator"""
    context = data.get("context", [])
    query = data.get("input", "")
    
    if isinstance(context, list):
        context_str = format_docs(context)
    else:
        context_str = str(context)
        
    return {
        "context": context_str,
        "input": query
    }

# Generator chain
generator_chain = (
    RunnableLambda(process_generator_input)
    | generator_prompt 
    | instruct_llm 
    | StrOutputParser()
)

# FastAPI app setup
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="RAG server using Langchain's Runnable interfaces",
)

# Add routes with configurable streaming
add_routes(
    app,
    instruct_llm,
    path="/basic_chat"
)

add_routes(
    app,
    retriever_chain,
    path="/retriever"
)

add_routes(
    app,
    generator_chain,
    path="/generator",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)