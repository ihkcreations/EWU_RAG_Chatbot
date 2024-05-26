from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOC_PATH = "docs"
DB_PATH = "chromadb"
embedding_model_name="nomic-embed-text"

loader = TextLoader('docs/ewu.txt', encoding = 'UTF-8')
documents = loader.load()

chunk_size=3900
chunk_overlap=500

text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

docs = text_splitter.split_documents(documents)
embeddings = OllamaEmbeddings(model = embedding_model_name)

print('| ---------------------------------------------------------|')
print(f'| Chunk size:{chunk_size}\n| Chunk overlap:{chunk_overlap}')
print(f'| {len(docs)} chunks are being embedded now, processing...')
vectordb = Chroma.from_documents(documents = docs, embedding=embeddings, persist_directory=DB_PATH)
vectordb.persist()

if vectordb:
    print(f"| Embeddings are generated with chunks: {len(docs)}")
    print('| ---------------------------------------------------------')


