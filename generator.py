import os
import fitz  # PyMuPDF for PDF reading
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder
import chromadb
from chromadb.utils import embedding_functions


# Initialize the ChatOpenAI model
def initialize_model(openai_key):
    try:
        return ChatOpenAI(api_key=openai_key,model="gpt-4o-mini")
    except Exception as e:
        st.error(f"Error initializing ChatOpenAI model: {e}")
        return None


# Convert PDF file to text
def from_pdf_to_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with fitz.open(file_path) as doc:
            text = "".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        raise Exception(f"Error reading PDF file: {e}")


# Create a vector store from document splits
def create_vector_store(splits):
    try:
        return InMemoryVectorStore.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )
    except Exception as e:
        raise Exception(f"Error creating vector store: {e}")

# Run the question-answer chain
def run_qa_chain(rag_chain, question, chat_history):
    try:
        # Invoke the chain and get the response
        ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
        
        # Ensure that the response contains an 'answer' key
        if "answer" not in ai_msg:
            raise KeyError("Expected key 'answer' not found in response.")
        
        # Append human question and AI answer to chat history
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=ai_msg["answer"]))
        
        # Return the AI answer
        return ai_msg["answer"]
    
    except Exception as e:
        raise Exception(f"Error running QA chain: {str(e)}")


# Function to recommend a book based on similarity search in ChromaDB
def recommend_book_from_chromadb(user_query, chromadb_client, embedding_function):
    try:
        results = chromadb_client.query(query_texts=[user_query], n_results=1)
        return f"Recommended book: {results['documents'][0]['title']}"
    except Exception as e:
        return f"Error during book recommendation: {e}"
class RetrievalQAChain:
    def __init__(self, retriever, qa_chain):
        self.retriever = retriever
        self.qa_chain = qa_chain

    def invoke(self, inputs):
        question = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        # Use the retriever to get relevant documents
        documents = self.retriever.retrieve(question, chat_history)

        # Use the QA chain to generate the answer based on the retrieved documents
        answer = self.qa_chain.generate_answer(question, documents)
        return {"answer": answer}

def create_retrieval_chain(retriever, qa_chain):
    return RetrievalQAChain(retriever, qa_chain)
class StuffDocumentsChain:
    def __init__(self, model, prompt, document_variable_name="context"):
        self.model = model
        self.prompt = prompt
        self.document_variable_name = document_variable_name

    def generate_answer(self, question, documents):
        # Stuff the documents into the prompt and create a query
        context = " ".join([doc.content for doc in documents])
        query = f"{self.prompt}: {question}\n{self.document_variable_name}: {context}"
        
        # Use the model to generate an answer
        answer = self.model.generate(query)
        return answer

def create_stuff_documents_chain(model, prompt, document_variable_name="context"):
    return StuffDocumentsChain(model, prompt, document_variable_name)
class HistoryAwareRetriever:
    def __init__(self, model, retriever, prompt):
        self.model = model
        self.retriever = retriever
        self.prompt = prompt

    def retrieve(self, question, chat_history):
        # Modify the retrieval logic to incorporate chat history
        history_context = " ".join([msg.content for msg in chat_history])
        query = f"{self.prompt}: {history_context} {question}"
        
        # Use the retriever to fetch relevant documents based on the query
        documents = self.retriever.retrieve(query)
        return documents

def create_history_aware_retriever(model, retriever, prompt):
    return HistoryAwareRetriever(model, retriever, prompt)