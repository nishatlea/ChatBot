import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone

# Initialize Pinecone with the correct API key and environment
pinecone_api_key = os.getenv('PINECONE_API_KEY')
environment = 'us-east-1'
pinecone.init(api_key=pinecone_api_key, environment=environment)

# Load documents from PDF file
file_path = "data/books/alice_in_wonderland.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

# Split documents into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Initialize OpenAI Embeddings
openai_api_key = os.getenv('OPENAI_API_KEY')
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key
)

# Initialize Pinecone instance
pc = Pinecone(
    pinecone_api_key=pinecone_api_key,
    index_name='serverless-index',
    embedding=embeddings
)

# Create index using Pinecone
index = Pinecone.from_documents(docs, embeddings, index_name='serverless-index', namespace='myspace')
