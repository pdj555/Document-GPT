import os
import constants

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index import download_loader


# OpenAI API Key
os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Load the data
documents = SimpleDirectoryReader("/Users/prestonjones/Library/CloudStorage/OneDrive-UniversityofOklahoma/cs/documentGPT/pdfs").load_data()
index = VectorStoreIndex.from_documents(documents)

# Pubmed Loader
PubmedReader = download_loader("PubmedReader")

loader = PubmedReader()
documents = loader.load_data(search_query='brain cancer')

# Query
query_engine = index.as_query_engine()
query_engine.query("what is brain cancer?")

'''
# Store data
index.storage_context.persist()

# Retrieve stored data
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir='./storage')
# load index
index = load_index_from_storage(storage_context)
'''