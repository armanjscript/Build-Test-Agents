from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id=session_id, connection_string="sqlite:///memory.db")


template = ChatPromptTemplate.from_messages([
    ('placeholder', "{history}"),
    ('human', "{prompt}")
])

llm = OllamaLLM(model="qwen2.5:latest", temperature=0.5, max_tokens=250)


session_id = "Arman"
st.title("How can I help you today?")
st.write("Enter your query below:")
session_id = st.text_input("Enter your name", session_id)


if st.button("Start all new conversation"):
    st.session_state.chat_history = []
    get_session_history(session_id).clear()
    

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])



chain = template | llm | StrOutputParser()


def invoke_history(chain, session_id, prompt):
    history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="prompt",
        history_messages_key="history"
    )
    
    for response in history.stream({"prompt": prompt}, config={"configurable":{"session_id": session_id}}):
        yield response


prompt = st.chat_input("Enter your query")


if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    
    with st.chat_message('user'):
        st.markdown(prompt)
    
    
    with st.chat_message('assistant'):
        streamResponse = st.write_stream(invoke_history(chain, session_id, prompt))
        
    st.session_state.chat_history.append({'role': 'assistant', 'content': streamResponse})
