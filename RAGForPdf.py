from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain.chains.retrieval_qa.base import RetrievalQA

# LLM setup
llm = OllamaLLM(model="qwen2.5:latest", temperature=0.5, max_tokens=250, num_gpu=1)

# Embedding setup
embeddings = OllamaEmbeddings(model="nomic-embed-text", num_gpu=1)  # Updated

# PDF loading
pdf1 = "./pdf1.pdf"
pdf2 = "./pdf2.pdf"
pdf3 = "./pdf3.pdf"

pdfFiles = [pdf1, pdf2, pdf3]

documents = []

for pdf in pdfFiles:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(documents)

# Vector store creation
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory="./langchain_chroma"
)


#Retrieving from the Persistant Vector Datastore
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 1}
)

# print(
# retriever.batch(
#     [
#         "What is mental health?",
#         "What are the mental disorders?"
#     ]
# )
# )

#Using Document Retrieval Manually
query = "What are the mental disorders?"
# retrieved_docs = retriever.get_relevant_documents(query)
# context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
# prompt_template = ChatPromptTemplate.from_template(
#     """
#     You are an AI Assistant. Use the following context to answer the question correctly. 
#     If you dont know the answer, just tell, i dont know.
    
#     context: {context}\n\n
#     question: {question}\n\n
#     AI answer:
    
#     """
# )

# chain = prompt_template | llm | StrOutputParser()

# response = chain.invoke({"context": context_text, "question": query})
# print(response)


#Using Langchain hub for Prompt
# prompt = hub.pull("rlm/rag-prompt")

# chain = prompt | llm | StrOutputParser()

# response = chain.invoke({"context": context_text, "question": query})
# print(response)


#Retrieving data using RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

response = qa_chain.invoke(query)

sources = set(doc.metadata.get("source", "Unknown") for doc in response["source_documents"])

print(response['result'])
print("\n 📚 Sources used: ")
for source in sources:
    print(f"-{source}")
