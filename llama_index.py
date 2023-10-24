import os
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

from llama_index import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
query_engine.query("what is brain cancer?")