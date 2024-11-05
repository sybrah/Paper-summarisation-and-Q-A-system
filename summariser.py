
import os
import fitz
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from transformers import BartForConditionalGeneration, BartTokenizer, PegasusForConditionalGeneration, PegasusTokenizer
from io import BytesIO
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@st.cache_resource
def load_model():
    try:
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
    except ImportError as e:
        st.error(f"Error loading model: {e}")
        return None, None
    return model, tokenizer

def from_pdf_to_text(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = "".join([page.get_text() for page in doc])
    return text

def chunk_text(text, max_tokens=1024):
    words = text.split()
    chunks = []
    chunk = []
    current_length = 0
    for word in words:
        current_length += len(word)
        chunk.append(word)
        if current_length >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            current_length = 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def summarize_by_chunking(text, model, tokenizer, max_length, min_length):
    chunks = chunk_text(text, max_tokens=1024) 
    summaries = []
    for chunk in chunks:
        inputs = tokenizer.encode(chunk, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs, 
            max_length=max_length,  
            min_length=min_length,  
            length_penalty=4.0,  
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)    
    return " ".join(summaries)

def summarize_full_text(text, model, tokenizer, max_length, min_length):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
            inputs, 
            max_length=max_length,  
            min_length=min_length,  
            length_penalty=4.0,  
            num_beams=4, 
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
