# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# def initialize_vectorstore():
#     return Chroma(
#         persist_directory="data/vector_db",
#         embedding_function=HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={"device": "cpu"}
#         )
#     )

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_vectorstore():
    return Chroma(
        persist_directory="data/vector_db",
        embedding_function=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}
        )
    )