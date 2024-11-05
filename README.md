# LLM: RAG for Paper Summarization, Q&A Chatbot, and Recommendation System

This project leverages state-of-the-art language models and database technologies to create a robust pipeline for summarizing research papers, answering user questions about submitted documents, and recommending related resources based on user queries. Below is a breakdown of the key components and technologies used.

## Project Overview
The project is designed to support users in managing, exploring, and learning from AI-related documents by:
1. Generating vector embeddings for fast document retrieval.
2. Summarizing document content.
3. Enabling a Q&A chatbot for document-specific inquiries.
4. Recommending relevant documents based on the user's interests and question themes.

## Project Demo

[Watch the video](C:\Users\brahe\Downloads\DEMO.mp4)

### Technologies Used
- **Sentence-BERT** for embedding generation
- **ChromaDB** for vector storage and similarity search
- **BART and Pegasus** for document summarization
- **GPT-4** in a Retrieval-Augmented Generation (RAG) pipeline for the Q&A chatbot

## Steps and Technology Details

### 1. Document Embedding with Sentence-BERT
   - **Purpose**: To convert documents into dense vector embeddings that capture semantic information, making them ideal for quick and relevant retrieval.
   - **Technology**: *Sentence-BERT (SBERT)*, a fine-tuned BERT-based model designed for producing sentence embeddings, was used to process and convert each AI-related document into a fixed-length vector.
   - **Storage**: The embeddings were stored in *ChromaDB*, a vector database optimized for similarity search. This storage choice ensures that related documents can be efficiently retrieved during recommendation.

### 2. Summarization of Uploaded Documents
   - **Purpose**: To provide users with a concise summary of uploaded documents, allowing them to choose between a full document summary or segmented, chunk-based summaries.
   - **Technology**:
     - *BART* and *Pegasus* models, both pre-trained on summarization tasks, were used here. BART excels in generating coherent, comprehensive summaries, while Pegasus is known for handling more technical or research-oriented documents.
   - **Summarization Modes**:
     - **Full-text Summarization**: Condenses the document into a single, coherent summary.
     - **Chunk-based Summarization**: Divides the document into meaningful sections and summarizes each section individually, useful for lengthy or complex documents.

### 3. Q&A System with Retrieval-Augmented Generation (RAG)
   - **Purpose**: To answer user-specific questions based on the content of their submitted document, ensuring that responses are directly relevant and grounded in the uploaded material.
   - **Technology**:
     - *GPT-4*, integrated into a RAG pipeline, was employed as the core of the Q&A system.
     - **RAG Pipeline**: Combines retrieval with generation. ChromaDB retrieves relevant portions of the document, and GPT-4 generates answers specifically based on the retrieved content. This allows the model to answer questions directly from the user’s document, ensuring that responses are accurate and context-aware.

### 4. Recommendation System Based on User Questions
   - **Purpose**: To suggest additional documents that align with the themes and subjects raised by the user through their questions.
   - **Technology**:
     - After users ask a minimum of five questions, the similarity search feature of ChromaDB is activated. This system analyzes the themes of the questions and finds documents with similar semantic content.
     - This similarity search is powered by the embeddings stored in ChromaDB, which allows it to locate documents that are relevant to the topics of the user's queries, enhancing the learning and exploration experience.

## Key Features
1. **Fast and Relevant Retrieval**: Leveraging Sentence-BERT embeddings stored in ChromaDB for rapid document retrieval.
2. **Flexible Summarization**: Allows users to select from full or chunk-based summaries, ensuring tailored document insights.
3. **Intelligent Q&A**: GPT-4-based RAG model that provides accurate, document-specific answers.
4. **Contextual Recommendations**: Uses the themes from user questions to recommend documents, making the system adaptable to individual learning paths.

