# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# vectorstore = Chroma(
#     persist_directory="data/vector_db",
#     embedding_function=HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"}
#     )
# )
# print(len(vectorstore.get()["ids"]))

from langchain_chroma import Chroma
vectorstore = Chroma(persist_directory="data/vector_db")
docs = vectorstore.get()['documents']
print(docs[0])  # Check if page_content is a string