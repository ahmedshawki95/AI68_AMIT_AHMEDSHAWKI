
import numpy as np
from flask import Flask, request, jsonify, render_template
import json 
import re 
import numpy as np 
import tensorflow as tf
import pickle
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model # type: ignore


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

#Define App
app = Flask(__name__, template_folder='templates')
#an object of class Flask which would do all the heavy lifting for us, like handling the incoming requests from the browser 
# and providing with appropriate responses (in this case our model prediction) in some nice form using html,css. 

# Load necessary data and models
data = json.loads(open(r'C:\Users\workstation\OneDrive - Alexandria University\AI\final project\final version\intents.json', 'r', encoding='utf-8').read())
words = pickle.load(open(r'C:\Users\workstation\OneDrive - Alexandria University\AI\final project\final version\model\words.pkl', 'rb'))
classes = pickle.load(open(r'C:\Users\workstation\OneDrive - Alexandria University\AI\final project\final version\model\classes.pkl', 'rb'))
model=load_model(r'C:\Users\workstation\OneDrive - Alexandria University\AI\final project\final version\model\chatbot_model.keras') # load model 

    

@app.route('/')
def home():
    return render_template('home.html')


# Function to preprocess the input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert input sentence to bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the class of the input sentence
def predict_class(sentence):
    print("predict_class",sentence)
    bow = bag_of_words(sentence)
    # Predict the intent probabilities
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Fetch the response based on intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I am sorry, I didn't understand that."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['label'] == tag:
            return random.choice(i['responses'])
    return "I am sorry, I don't have a response for that."


@app.route('/',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    message=request.form.get("user")
    print("message:",message)
    intents_list = predict_class(message)
    response = get_response(intents_list, data)


    return render_template('home.html', prediction_text='{} '.format(response))



if __name__ == "__main__":
    app.run(debug=True)
