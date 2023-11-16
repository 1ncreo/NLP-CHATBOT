import uvicorn
from fastapi import FastAPI

# ML Pkg
import numpy as np
import random
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Vectorizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    
#data_vectorizer = open("nlp-chatbot/data.pkl","rb")
#query_cv = joblib.load(data_vectorizer)

FILE = "nlp-chatbot/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jarvis"

bot_output = "Let's chat! (type 'quit' to exit)"
# Models


#init app
app = FastAPI()

#Routes
@app.get('/')
async def index():
    return {"text":"Hello API Masters"}

@app.get('/items/{bot_input}')
async def get_items(bot_input):
    return {"bot_input":bot_input}

#ML Aspect
@app.get('predict/{bot_input}')
async def predict(bot_input):

    while True:
        # sentence = "do you use credit cards?"
        # sentence = input("You: ")
        sentence = bot_input
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    #print(f"{bot_name}: {random.choice(intent['responses'])}")
                    bot_output = {random.choice(intent['responses'])}
                    return {"bot_input":bot_output}
        else:
            #print(f"{bot_name}: I do not understand...")
            bot_output = "I do not understand..."
            return {"bot_input":bot_output}


if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1", port=8000)

