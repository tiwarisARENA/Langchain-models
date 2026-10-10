from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model='gpt-4', temperature=0.7, max_completion_token=10) # Initialize the ChatOpenAI model with a temperature of 0.7.
#temperatur and max_completion_token are optional parameters that can be adjusted based on your requirements. and also there are other parameters
# that can be adjusted based on your requirements.
#temperature controls the randomness of the model's responses.
# A higher temperature (e.g., 0.9) will make the output more random, while a lower temperature (e.g., 0.2) will make it more focused and deterministic.
# max_completion_token limits the maximum number of tokens in the model's response. In this case, it is set to 10,
# which means the model will generate a response with a maximum of 10 tokens.
result = model.invoke("what is the distance between paris and london?")
print(result.content) # Print the content of the response from the language model

