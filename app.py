import os
import streamlit as st
import fitz
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
from Scripts.generator import initialize_model , from_pdf_to_text,create_vector_store,run_qa_chain,recommend_book_from_chromadb
from Scripts.summariser import from_pdf_to_text , load_model ,  summarize_by_chunking , summarize_full_text , chunk_text

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "******")
CHROMA_DATA_PATH = r"./embeddings_database"
DATA_DIR = r"./data"
COLLECTION_NAME = "AI_related_documents"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBED_FUNCTION, metadata={"hnsw:space": "cosine"})


def main():
    # Streamlit page configuration
    st.set_page_config(page_title="Summarization & Q&A System", page_icon="🤖", layout="wide")

    st.title("🤖 Summarization & Q&A System")
    st.markdown(
        """
        Welcome to the **Summarization & Question-Answering** system. 
        Upload your PDF document, summarize it with powerful models, and ask questions from the document!
        """
    )

    # Step 1: PDF Upload and Summarization
    st.header("Step 1: Summarization")
    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

    if uploaded_file:
        # Extract full text from PDF
        full_text = from_pdf_to_text(uploaded_file)
        
        # Display extracted text in a collapsible section
        with st.expander("📄 View Extracted Text", expanded=False):
            st.text_area("Extracted Text from PDF", full_text[:5000], height=300)  # Show only the first 5000 chars for readability

        # Layout for model and method selection
        col1, col2 = st.columns(2)
        with col1:
            options_model = ["Bart", "Pegasus"]
            chosen_model = st.selectbox('🛠️ Choose a summarization model', options_model)
        
        with col2:
            options_method = ["Chunking_text", "Full_text"]
            chosen_method = st.selectbox('🧩 Choose a summarization method', options_method)
        
        # Load the selected model
        model, tokenizer = None, None
        if chosen_model == "Bart":
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        elif chosen_model == "Pegasus":
            model, tokenizer = load_model()  # Implement this function to load Pegasus

        # Summarization parameters
        max_length = st.slider("Max summary length", min_value=50, max_value=500, value=100)
        min_length = st.slider("Min summary length", min_value=10, max_value=100, value=30)

        if st.button("📝 Summarize"):
            with st.spinner("Summarizing the document..."):
                if chosen_method == "Chunking_text":
                    summary = summarize_by_chunking(full_text, model, tokenizer, max_length, min_length)
                else:
                    summary = summarize_full_text(full_text, model, tokenizer, max_length, min_length)

                st.success("✅ Summarization complete!")
                st.subheader("📜 Summary:")
                st.write(summary)

                # Allow downloading of summary
                summary_bytes = BytesIO(summary.encode())
                st.download_button("⬇️ Download Summary as Text File", data=summary_bytes, file_name="summary.txt", mime="text/plain")

        st.divider()
        
    if uploaded_file:
        # Step 2: Q&A Interface
        st.header("Step 2: Ask Questions")
        st.markdown("Use the summarized text or entire document to ask questions.")

        # Initialize OpenAI model for Q&A
        model = initialize_model(os.getenv("OPENAI_API_KEY"))
        if model is None:
            st.error("❌ Error: Model initialization failed.")
            return

        # Process PDF and create vector store
        document = Document(page_content=full_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents([document])

        try:
            vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        except Exception as e:
            st.error(f"❌ Error creating vector store: {e}")
            return

        retriever = vectorstore.as_retriever()

        # System prompt for Q&A
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )

        # Create prompt and chain for Q&A
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(model, retriever, prompt)
        question_answer_chain = create_stuff_documents_chain(model, prompt, document_variable_name="context")
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chat_history = []
        question_counter = 0
        if question_counter < 2:
            question = st.text_input("💬 Ask a question (You can ask up to 5 questions):")
            if st.button("💡 Get Answer"):
                if question:
                    try:
                        with st.spinner("Retrieving answer..."):
                            answer = run_qa_chain(rag_chain, question, chat_history)
                            st.success(f"Answer: {answer}")
                            question_counter += 1

                            # Recommend a book after 5 questions
                            if question_counter == 2:
                                st.info("📚 Recommending a book based on your questions...")
                                recommendation = recommend_book_from_chromadb(" ".join([q.content for q in chat_history]), chromadb_collection, EMBED_FUNCTION)
                                st.write(recommendation)
                    except Exception as e:
                        st.error(f"❌ Error running QA chain: {e}")

        st.write("🗣️ Chat History:", chat_history)

if __name__ == "__main__":
    main()