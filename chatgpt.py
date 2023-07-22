import os
import sys

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_KEY')

loader = DirectoryLoader("./work-policies", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
query = sys.argv[1]

print(index.query(query, llm=ChatOpenAI(temperature=0)))