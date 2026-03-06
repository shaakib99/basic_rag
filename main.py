from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

load_dotenv()


loader = DirectoryLoader(path = './data/', glob="*.txt", loader_cls= TextLoader, recursive=True)
documents: list[Document] = loader.load()

text_splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=200)
splitted_documents = text_splitter.split_documents(documents)

transformer = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

store = InMemoryVectorStore.from_documents(splitted_documents, transformer)

query = '2nd Text file'

result = store.similarity_search(query, k = 1)
print(result)
