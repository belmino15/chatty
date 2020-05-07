import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

training = pickle.load(open('training.pkl','rb'))

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