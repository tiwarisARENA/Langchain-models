from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
document = [
    "silicon valley is the hub of startups in the world",
    "and nyc is the hub of finance and hft firms"
]

vector = embedding.embed_documents(document)
print(str(vector))