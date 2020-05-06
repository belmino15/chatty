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

# criando base de treinamento
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = [] # inicializando (e zerando) bag
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # bag terá o mesmo tamanho de words, representando (em formato de vetor) as palavras contidas em "doc"
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # a saida é composta por 0 e 1, indicando a presença da palavra de "words" no no "doc"
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row]) # algo do tipo - ([0,1,0,1,1,0,0,1,0,1], "saudacao")

# Misturando os dados e transformando em vetor do numpy
random.shuffle(training)
training = np.array(training)

# separando parâmetros das saida. X - parâmetros (vetor de 0/1 indicando a presença das palavras), Y - saidas (intenção)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Criação do modelo - multi-class softmax classification (MLP, proposto na documentação do keras)
# https://keras.io/getting-started/sequential-model-guide/ - Ver "Multilayer Perceptron (MLP) for multi-class softmax classification"
# Modelo tera 128 perceptrons na primeira camada, 64 na seungda e a ultima igual ao número de saídas (seguindo o modelo de ativação da saída)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Modelo de compilação - Definindo a "processo de aprendizado" 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(  loss='categorical_crossentropy', # função de aptidão
                optimizer=sgd, # escolha do metodo de otimização
                metrics=['accuracy']) # função metrica, para avaliar a performance do modelo - recomendado 'accuracy'

#treinando e salvando o modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")