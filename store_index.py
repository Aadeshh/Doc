from src.helper import load_pdf_file, text_split, downloadHuggingFaceModel
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_file("/Users/aadesh/Downloads/Doc/Doc/data")
text_chunks = text_split(extracted_data)
embeddings = downloadHuggingFaceModel()




pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

pc.create_index(index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
) 