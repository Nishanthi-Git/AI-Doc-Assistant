import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Updated import for the modern LangChain structure
from langchain_classic.chains import RetrievalQA

# ---------------- UI Setup ----------------
st.set_page_config(page_title="AI Assistant", layout="centered")
st.title("📄 Junior Developer Technical Demo")
st.markdown("AI Doc Assistant - Analyze technical manuals with automated AI workflows.")

# ---------------- API Key Sidebar ----------------
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Add the Demo Mode toggle here
with st.sidebar:
    demo_mode = st.checkbox("Enable Demo Mode (No API required)")

if api_key or demo_mode: # Allow app to run if Demo Mode is on
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    uploaded_file = st.file_uploader("Upload a PDF manual", type="pdf")

    if uploaded_file:
        # Save file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Only run the heavy processing if NOT in demo mode
        if not demo_mode:
            with st.spinner("Processing document..."):
                loader = PyPDFLoader("temp.pdf")
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.success("Document analyzed and ready!")
        else:
            st.success("Demo Mode Active: PDF recognized (Simulated Analysis)")

        # Chat interface
        question = st.text_input("Ask a question about the document:")

        if question:
            if demo_mode:
                # ---------------- DEMO LOGIC ----------------
                import time
                with st.spinner("Demo Mode: AI is searching the manual..."):
                    time.sleep(2) # Simulate thinking time
                    st.write("### Answer:")
                    # This provides a smart-looking answer for your Bosch manual
                    st.info("Based on the GSR 18V-55 Professional manual, this tool features a brushless motor with a maximum torque of 55 Nm. It is designed for drilling in wood (up to 35mm) and steel (up to 13mm).")
            else:
                # ---------------- REAL API LOGIC ----------------
                with st.spinner("AI is searching the manual..."):
                    try:
                        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=vectorstore.as_retriever()
                        )
                        answer = qa_chain.invoke({"query": question})
                        st.write("### Answer:")
                        st.info(answer["result"])
                    except Exception as e:
                        st.error(f"Quota/API Error: {e}")
                        st.warning("Switch to 'Demo Mode' in the sidebar to continue the presentation.")
else:
    st.warning("Please enter your OpenAI API key or enable Demo Mode in the sidebar to start.")
    st.divider()
    st.caption("Note: This is an AI prototype. Please verify technical answers with official Bosch manuals.")

 # ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    st.caption("👤 **Developer:** Nishanthi")
with f_col2:
    st.caption("🛠️ **Stack:** Python 3.11, LangChain, FAISS")
with f_col3:
    st.caption("⚠️ For Interview Purpose Only")   
