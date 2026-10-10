from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
model = OpenAI(model='gpt-4')  # Initialize the ChatOpenAI model
result = model.invoke("where is italy in europe?")  # Invoke the language model with the input question
print(result.content)  # Print the content of the response from the language model