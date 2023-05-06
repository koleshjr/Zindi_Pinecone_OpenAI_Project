import pinecone
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from consts import llm_model_cohere, llm_model_openai, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

#loading the txt files
loader = DirectoryLoader(
    "output",
    glob = "**/*.txt",
    loader_cls = TextLoader,
)

documents = loader.load()

#splitting the text into chunks
text_splitter = CharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap= 32,

)

texts = text_splitter.split_documents(documents)


# persist_directory = 'db' #save on money
embeddings = CohereEmbeddings(model = llm_model_cohere,cohere_api_key=cohere_api_key)

pinecone.init(
    api_key=pinecone_api_key , # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT,  # next to api key in console
)


vectordb= Pinecone.from_documents(texts, embeddings, index_name=PINECONE_INDEX_NAME)
