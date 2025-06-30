from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vectorstore = Chroma(
    persist_directory="data/vector_db",
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
)
print(len(vectorstore.get()["ids"]))