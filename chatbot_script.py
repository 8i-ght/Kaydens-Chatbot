# BEFORE RUNNING THIS BOT REFER TO THE README FILE FOR INSTRUCTIONS
# This is a chatbot that people can ask simple questions to learn more about me.
# I will be using a question answering AI model from the Hugging Face Transformers library to do this. 

# Import the dataset
import json
from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the dataset
with open('kayden_data.json', 'r') as file:
    kayden_data = json.load(file)

# Convert the hobbies list to a string
# The chatbot has more confidence on a string rather than list 
hobbies_str = ', '. join(kayden_data['hobbies'])

# Create the context
context = f"""
Kayden's name is {kayden_data['name']}.
Kayden is {kayden_data['age']} years old.
Kayden is a {kayden_data['gender']}.
Kayden's race is {kayden_data['race']}.
Kayden's major is {kayden_data['major']}.
Kayden is a {kayden_data['year']}.set
Kayden's hobbies are {hobbies_str}.
Kayden doesn't have a favorite food.
Kayden's job is {kayden_data['job']}.
Kayden goes to school at {kayden_data['school']}.
Kayden's hometown is {kayden_data['hometown']}.
Kayden's email is {kayden_data['email']}.
Kayden's phonenumber is {kayden_data['phonenumber']}.
Kayden's socials are found at {kayden_data['socials']}.
Kayden's projects are found at {kayden_data['projects']}.
Kayden's resume is {kayden_data['resume']}.
"""

# Create the question answering pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='distilbert-base-uncased', max_seq_len=384, doc_stride=128)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Get the answer from the pipeline
    response = qa_pipeline(question=question, context=context)
    answer = response['answer']
    score = response['score']

    # Set a confidence threshold
    confidence_threshold = 0.24
    if score >= confidence_threshold:
        return jsonify({'answer': answer, 'score': score})
    else:
        return jsonify({'answer': "I'm sorry, I do not have an answer to that question.", 'score': score})

if __name__ == '__main__':
    app.run(debug=True)