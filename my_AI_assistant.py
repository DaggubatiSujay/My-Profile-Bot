import streamlit as st
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Function to initialize Pinecone and LangChain components
def initialize_components(pinecone_api_key, openai_api_key):
    pc = pinecone.Pinecone(api_key=pinecone_api_key, environment = 'us-east-1')

    index_name = 'resume-bot'
    
    index = pc.Index(index_name)

    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vector_store = PineconeVectorStore(index=index, embedding=embedding_function, text_key='text')

    qa_prompt = PromptTemplate.from_template(
        "Answer the question based on the given context: {context}\nQuestion: {question}\nAnswer:"
    )

    qa_chain = load_qa_chain(OpenAI(api_key=openai_api_key), prompt=qa_prompt)

    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(api_key=openai_api_key, max_tokens=1024),
        retriever=vector_store.as_retriever()
    )
    
    return retrieval_chain

# Function to get response from the chatbot
def get_response(retrieval_chain, prompt, chat_history):
    response = retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    chat_history.append((prompt, response['answer']))
    return response['answer']

# Streamlit app
st.title("Chatbot with Pinecone and OpenAI")

# Step 1: Enter API keys
pinecone_api_key = "53cd4cba-f9e3-4d50-a7f1-fca11d55cb79"
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if st.button("Initialize Chatbot"):
    if not openai_api_key:
        st.error("Please provide OpenAI API key.")
    else:
        st.session_state['retrieval_chain'] = initialize_components(pinecone_api_key, openai_api_key)
        st.session_state['chat_history'] = []

# Step 2: Chat interface
if 'retrieval_chain' in st.session_state:
    st.subheader("Chat with the Bot")
    prompt = st.text_input("Enter your prompt:")
    if st.button("Get Response"):
        if prompt:
            response = get_response(st.session_state['retrieval_chain'], prompt, st.session_state['chat_history'])
            st.text_area("Response:", value=response, height=200)
        else:
            st.error("Please enter a prompt.")

    st.subheader("Chat History")
    for prompt, response in st.session_state['chat_history']:
        st.write(f"User: {prompt}\nBot: {response}")