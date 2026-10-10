from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embeddings = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
document = [
    "zurich is more expensive than nyc",
    "but still nyc is far ahead than zurich in terms of no. of startups",
    "however zurich is beautiful and also has a good quality of life",
    "nyc is a city that never sleeps and has a vibrant culture",
    "zurich is known for its high standard of living and cleanliness",
]
query = "which city is better for startups, zurich or nyc?"

doc_embeddings = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)
# this will print the similarity scores between the query and each document in the list.
# example output: [[0.12345678 0.23456789 0.34567890 0.45678901 0.56789012]]
# this is a 2d list.
print(similarities) 
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
# this will give 1d list of similarity scores between the query and each document in the list.
# example output: [0.12345678 0.23456789 0.34567890 0.45678901 0.56789012]
# to find the index of highest similarity score, we need to sort the list but this will change the order of the list.
# so will use enumerate to get the index of each element in the list and then sort the list based on the similarity score.
similarities = list(enumerate(similarities))
# output : [(0, 0.12345678), (1, 0.23456789), (2, 0.34567890), (3, 0.45678901), (4, 0.56789012)]
index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]
# this will sort the list based on the similarity score in ascending order.
# example output: [(0, 0.12345678), (1, 0.23456789), (2, 0.34567890), (3, 0.45678901), (4, 0.56789012)]
# to get the index of the highest similarity score, we can use the last element of the sorted list.
# and this index and score will be the index and score of the document that is most similar to the query.

print(query)
print(document[index])
print("similarity score: ", score)
