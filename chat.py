import random
import json

import torch
import numpy as np
import torch
import torch.nn as tnn
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

all_words = data['all_words']
model_state = data["model_state"]
tags = data['tags']
output_size = data["output_size"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]



class NeuralNet(tnn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = tnn.Linear(input_size, hidden_size) 
        self.l2 = tnn.Linear(hidden_size, hidden_size) 
        self.l3 = tnn.Linear(hidden_size, num_classes)
        self.relu = tnn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

def get_response(msg):
    sentence = tokenize(msg)
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
                return random.choice(intent['Respostas'])
    
    return "Não entendi poderia ser mais expecifico..."


if __name__ == "__main__":
    print("Debug Chat SOMENTE PARA TESTE")
    while True:
        #a frases que o usuario digita"
        sentence = input("Voce: ")
        if sentence == "Sair":
            break

        resp = get_response(sentence)
        print(resp)

