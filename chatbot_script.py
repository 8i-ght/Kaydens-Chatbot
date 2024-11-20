# This is a chatbot that people can ask simple questions to learn more about me.
# I will be user a question answering AI model from the Hugging Face Transformers library to do this. 
# Users can refer to the README.md file for more information on how to use this chatbot.

# Import the dataset
import json
from transformers import pipeline

# Load the dataset
with open('kayden_data.json', 'r') as file:
    kayden_data = json.load(file)

# Create the context
context = f"""
Kayden's name is {kayden_data['name']}.
Kayden is {kayden_data['age']} years old.
Kayden's major is {kayden_data['major']}.
Kayden is a {kayden_data['year']}.set
Kayden's hobbies are {kayden_data['hobbies']}.
Kayden's job is {kayden_data['job']}.
Kayden goes to school at {kayden_data['school']}.
Kayden's hometown is {kayden_data['hometown']}.
Kayden's email is {kayden_data['email']}.
Kayden's phonenumber is {kayden_data['phonenumber']}.
Kayden's socials are found at {kayden_data['socials']}.
Kayden's projects are found at {kayden_data['projects']}.
"""

# Create the question answering pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='distilbert-base-uncased', max_seq_len=384, doc_stride=128)

# Define the chatbot function
def chatbot(prompt):
    print("Hello! I am a chatbot. You can ask me questions about Kayden.")
    response = qa_pipeline(question=prompt, context=context)
    answer = response['answer']

    # Check if the question is relevent to the context
    if answer.lower() in context.lower() and len(answer) > 0:
        return answer
    else: 
        raise ValueError("I'm sorry, I do not have an answer to that question. Please ask me something else.")


while True:
    user_input = input("Ask a question (or type 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    try:
        print(chatbot(user_input))
    except ValueError as e:
        print(e)
