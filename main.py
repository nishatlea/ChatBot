import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from pinecone import ServerlessSpec

file_path = "data/books/alice_in_wonderland.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

from langchain_openai import OpenAIEmbeddings

openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key
)

from langchain_pinecone import Pinecone

environment = 'us-east-1'
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone = Pinecone(api_key=pinecone_api_key)
# indexes = pinecone.list_indexes()
# print(indexes.names())
pinecone.create_index(
    name="serverless-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

pc = Pinecone(
    pinecone_api_key=os.getenv('PINECONE_API_KEY'),
    index_name='my_index',
    embedding=embeddings
)

index = Pinecone.from_documents(docs, embeddings, index_name='my_index')
retriever = index.as_retriever(search_type='similarity', search_kwargs={"k": 2})

