import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

words=[]
classes = []
documents = []
ignore_words = ['?', '!']

data_file = open('intents.json').read()
intents = json.loads(data_file)

''' EXEMPLO DE TOKENIZE
sentence = "At eight o'clock on Thursday morning, Arthur didn't feel very good."
print(nltk.word_tokenize(sentence))
OUTPUT = ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning', ',', 'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']
'''

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # isolando as palavras da frase
        w = nltk.word_tokenize(pattern) # ver explicação acima
        words.extend(w)

        # criando os documentos com as palavras isoladas
        documents.append((w, intent['tag']))

        # criando as classes, cada intenção será uma classe
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words] # lematizando as palavras (gerar a raiz da palavra)
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))