import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter





def get_text_chunks_langchain(context):
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.schema.document import Document
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
    text_chunks = [Document(page_content=x) for x in text_splitter.split_text(context)]
    return text_chunks


context = """Profile of ABC Logistics ABC Logistics is one of India’s
             premier Express Distribution and Supply Chain Management
               companies, committed to delighting customers
         with seamless, end-to-end logistics solutions backed
           by future-ready digital tools and technology. 
           Founded in 1989, with renowned expertise in surface
             and air express as well as customized solutions
               for retail and MSME sectors, ABC Logistics gives 
               businesses the added advantage of an unmatched network.
                 ABC’s coverage spans the whole of India, 
                 bolstering businesses with access
                   to more than 19,800 PIN Codes and 735 of India’s 739 districts. 
            After strategically acquiring ABC Logistics in 2020, 
            Allcargo Logistics is now the promoter and 
            the single largest shareholder of ABC Logistics.
            As part of Allcargo Group, ABC Logistics is 
            uniquely equipped to offer truly end-to-end integrated 
            logistics with services across diverse verticals 
            that include International Supply Chain, CFS-ICD, 
            Contract Logistics, Logistics Parks, and more. 
            Further, businesses aspiring to broaden their horizons 
            can tap into a global network operating in 180 countries. 
            With a digital-first approach, plethora of 
            tools like digital payment modes, enterprise-wide ERP systems, 
            ABC Logistics Genie chatbot on WhatsApp, etc. 
            ABC looks ahead to delighting customers and enabling swift, 
            safe and timely deliveries, every single time. 
            As a responsible corporate citizen, 
            ABC Logistics is conscious of its environmental 
            impact and abides by its core value of ‘Care for Environment 
            and Society’ to adhere to world-class Environmental Social
              and Governance (ESG) standards and contribute to a better world."""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

max_tokens = 20
temperature = .4
text_chunks = get_text_chunks_langchain(context)

embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
embeddings = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={'device': 'cpu'})
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
retriever = vector_store.as_retriever()



# Working ok with following phi3 conf
from langchain_community.chat_models import ChatOllama
#from langchain_community.llms import Ollama

llm = ChatOllama(model="phi3", temperature=0.2, num_predict=30)




"""
# working with low accuracy
from langchain_community.llms import LlamaCpp
n_gpu_layers = 1
n_batch = 512
llm = LlamaCpp(model_path="./tt/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
               n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    temperature = 0,
    max_tokens = 30,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=False,)

"""



"""
# NEED TO FIX
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-gemma-v0.1")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-gemma-v0.1", torch_dtype=torch.float16, low_cpu_mem_usage=True)

tokenizer.save_pretrained("zephyr-7b-gemma-v0.1-tokenizer")
model.save_pretrained("zephyr-7b-gemma-v0.1-model", max_shard_size="1000MB")



llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": False},
    tokenizer_name="zephyr-7b-gemma-v0.1-tokenizer",
    model_name="zephyr-7b-gemma-v0.1-model",
    tokenizer_kwargs={"max_length": 2048},
    model_kwargs={"torch_dtype": torch.float16}
)

"""

'''
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
'''

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

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

session_id = "abc123"

while True:
    query = input("Type here...#######>>> ")
    result = conversational_rag_chain.invoke(
    {"input": query},
    config={
        "configurable": {"session_id": session_id}
    },  # constructs a key "abc123" in `store`.
    )
    print("----------------------------")
    #print("HISTORY :: ", result['chat_history'])
    print("RESULT :: ", result['answer'])
    print("------------------------------\n")