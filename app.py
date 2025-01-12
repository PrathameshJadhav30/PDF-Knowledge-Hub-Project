import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google GenAI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key is not configured! Please check your .env file.")
    st.stop()
genai.configure(api_key=api_key)

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save the vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create the conversational chain for Q&A
def get_conservational_chain(retriever):
    prompt_template = """Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details.
    If the answer is not in the provided context, just say, "Answer is not available in the context."
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load the vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = new_db.as_retriever()

        # Initialize the conversational chain
        chain = get_conservational_chain(retriever)

        # Pass the user question to the chain
        response = chain.run(query=user_question)
        
        # Display the response
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main application logic
def main():
    # Page config and title
    st.set_page_config(page_title="PDF Knowledge Hub", layout="wide")
    st.title("📚 PDF Knowledge Hub: Ask Anything From Your PDFs")
    
    st.markdown("""
    ## Interact with Your PDF Documents

    Upload your PDF files and ask detailed questions about their content.  
    You can upload multiple files, and the system will process them to allow you to inquire about the text contained within.
    """)

    # Sidebar for file upload and process button
    with st.sidebar:
        st.header("PDF Upload & Processing")
        pdf_docs = st.file_uploader("Choose Your PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs... Please wait."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully! You can now ask questions.")
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

    # Main interface for user question input
    user_question = st.text_input("What do you want to know about the PDFs?")
    if user_question:
        with st.spinner("Fetching response..."):
            user_input(user_question)

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by Prathamesh Jadhav | [Instagram](https://www.instagram.com/prathamesh_jadhav_30/) | [GITHUB](https://github.com/PrathameshJadhav30) | [Linkedin](https://www.linkedin.com/in/prathamesh-jadhav-b30490259/)")

if __name__ == "__main__":
    main()
