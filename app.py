import os  
import streamlit as st  
from openai import OpenAI
from PyPDF2 import PdfReader  
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS  
from langchain_classic.chains import RetrievalQA
from langchain_classic.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from react_agent import calculator_tool

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key="your-api-key-here")

# if not API_KEY:
#     st.error("Google API key is missing! Set it in your .env file.")
# else:
#     from google.generativeai import configure
#     configure(api_key=API_KEY)


def get_pdf_docs_with_metadata(pdf_docs):
    """
    Extracts text and metadata from uploaded PDF files.
    """
    docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                docs.append({
                    "text": page_text,
                    "metadata": {
                        "source": pdf.name,
                        "page": i + 1
                    }
                })
    return docs


def get_text_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """
    Splits extracted documents into smaller chunks for efficient processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    split_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            split_docs.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })
            
    return split_docs

def get_vector_store(docs):
    """
    Converts text chunks into embeddings and stores them using FAISS.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    texts = [doc["text"] for doc in docs]
    metadatas = [doc["metadata"] for doc in docs]
    
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")


def load_vector_store():
    """
    Loads FAISS vector store if it exists.
    """
    if os.path.exists("faiss_index"):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    return None


def handle_user_input(user_question):
    """
    Processes user input using a ReAct agent that supports multi-step retrieval.
    """
    vector_store = load_vector_store()
    if not vector_store:
        st.error("No processed PDF found. Please upload and process a PDF first.")
        return

    def search_documents(query: str) -> str:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        
        context = ""
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            context += f"[Source: {source}, Page: {page}]\n{doc.page_content}\n\n"
        return context

    search_tool = Tool(
        name="DocumentSearch",
        func=search_documents,
        description="Useful for searching information from the uploaded PDF documents. Input should be a specific search query."
    )

    tools = [search_tool, calculator_tool]
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
    
    prompt = PromptTemplate.from_template(template)
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_question})
        
    st.write("**Reply:**", response.get("output", response))


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Chat with PDFs using RAG")
    st.header("Chat with PDFs")

    user_question = st.text_input("Ask a question about your PDF files:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    docs = get_pdf_docs_with_metadata(pdf_docs)
                    text_chunks = get_text_chunks(docs)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
