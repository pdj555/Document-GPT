import os 
import sys 

import constants
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PubMedLoader

os.environ["OPENAI_API_KEY"] = constants.APIKEY

query = input("Enter your query: ")

#loader = TextLoader('data.txt')
#loader = DirectoryLoader("/Users/prestonjones/OneDrive - University of Oklahoma/cs/LangChain/documentGPT/pdfs/")
loader = PubMedLoader("chatgpt")

docs = loader.load()


index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI()))







'''
import os
import sys

import constants
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
import logging
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


os.environ["OPENAI_API_KEY"] = constants.APIKEY

loader = UnstructuredPDFLoader(
	“4401757.pdf”, mode=”elements”, strategy=”fast”,
) 
docs = loader.load()


# Split 
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)


# Store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())


# Retrieve
question = "What is brain cancer?"
docs = vectorstore.similarity_search(question)
len(docs)


svm_retriever = SVMRetriever.from_documents(all_splits,OpenAIEmbeddings())
docs_svm=svm_retriever.get_relevant_documents(question)
len(docs_svm)


# Logging
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(),
                                                  llm=ChatOpenAI(temperature=0))
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)


# Generate
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

from langchain.schema.runnable import RunnablePassthrough
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt 
    | llm 
)

rag_chain.invoke("What is brain cancer?")


template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | rag_prompt_custom 
    | llm 
)

rag_chain.invoke("What is brain cancer?")

'''