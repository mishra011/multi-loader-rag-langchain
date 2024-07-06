import os
import streamlit as st
import tempfile   # temporary file

# my_app.py
st.title("My Streamlit App")
st.write("Welcome to my interactive dashboard!")

if 'data' not in st.session_state:
    st.session_state['data'] = []
    

#st.button("Reset", type="primary")
if st.button("Reset"):
    st.session_state['data'] = []
    uploaded_files = []
    url = None
    st.write("Cache Cleaned")
# else:
#     st.write("Goodbye")



def saver(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    return tmp_file_path

if True:
    with st.form(key="UPLOAD"):
        #user_input = st.text_input("Enter text here:")
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        for uploaded_file in uploaded_files:
            print("UPLOADED FILE :: ", uploaded_file)
            ppath = saver(uploaded_file)
            st.session_state['data'].append(("file", ppath))

    
    # Create the text input and submit button within a form

if True: 
    with st.form(key="url_input_form"):
        url = st.text_input("Enter URL here:")
        submit_button = st.form_submit_button(label="Submit")

    # Check if the submit button was clicked
    if submit_button:
        # Validate user input (optional)
        if url:
        # Append the input to the list
            st.session_state['data'].append(("web",url))
            # Clear the text input field
            st.session_state["user_input"] = ""  # Use session state to store temporary data
        else:
            st.warning("Please enter some text before submitting.")

st.write(st.session_state['data'])
    




def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import textwrap

from langchain.llms import Ollama


# Loading llama3-7b from Ollama
llm = Ollama(model="llama3", temperature=0)

# Loading the Embedding Model


loader = None
text = []
if st.button("Train"):
    for item in st.session_state['data']:
    
        if item[0] == "file":
            if item[1].endswith(".pdf"):
                loader = PyPDFLoader(item[1])
                text += loader.load()


if text:
    text_splitter = RecursiveCharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)
    embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
    embeddings = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    retriever = vector_store.as_retriever()

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

if True:
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


