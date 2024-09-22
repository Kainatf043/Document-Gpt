import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import docx  # For handling .docx files

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded documents
def get_document_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        elif uploaded_file.name.endswith('.txt'):
            text += str(uploaded_file.read(), 'utf-8')  # Read as a text file

        # Debugging log for file processing
        st.write(f"Processed file: {uploaded_file.name} with length {len(text)} characters")
    
    return text

# Function to split the text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    st.write(f"Generated {len(chunks)} chunks of text.")  # Debugging log for chunk splitting
    return chunks

# Function to create and save the vector store for text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.write("Vector store created and saved.")  # Debugging log for vector store

# Function to create a conversational chain using Gemini
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say,
    'answer is not available in the context'. Do not provide an incorrect answer.

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide answers based on uploaded files
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    if not docs:
        st.error("No relevant context found in the documents for the question.")
    else:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])

# Main function to render the Streamlit app
def main():
    st.set_page_config("Chat with Documents using Gemini💁")
    st.header("Chat with Documents using Gemini💁")

    # Sidebar for uploading documents
    with st.sidebar:
        st.title("Menu:")
        uploaded_docs = st.file_uploader(
            "Upload your PDF, DOCX, or TXT Files", 
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if uploaded_docs:
                with st.spinner("Processing..."):
                    # Extract and process text from uploaded files
                    raw_text = get_document_text(uploaded_docs)
                    if raw_text.strip() == "":
                        st.error("No text could be extracted from the uploaded files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing completed!")
            else:
                st.error("Please upload at least one document.")

    # Input box for user's question
    user_question = st.text_input("Ask a Question from the Documents")
    
    if st.button("Answer the Query"):
        if user_question:
            with st.spinner("Searching for the answer..."):
                user_input(user_question)
        else:
            st.error("Please enter a question to get an answer.")

# Run the app
if __name__ == "__main__":
    main()
