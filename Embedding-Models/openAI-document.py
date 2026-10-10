from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
document = [
    "zurich is more expensive than nyc",
            "but still nyc is far ahead rhan zurich in terms of no. of startups",
            "however zurich is beautiful and also ahave a good quality of life"
            ]
result = embeddings.embed_documents(document)
print(str(result))

