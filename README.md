# Chatty - ChatBot em python
Este é um projeto de um bot simples para aprendizado. O bot utilizará de pacote de linguagem natural de máquina (nltk) e de aprendizado de máquina (keras), versões:

	python = 3.8.2
	tensorflow = 2.2.0
	keras = 2.3.1
	nltk = 3.5
	numpy = 1.18.4

Referência
: O projeto tomou como base o chatbot proposto em: https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44

## Próximos objetivos:
- Utilizar mongoDB para arquivar as intenções (intents.json);
- Separar o arquivo de lemantizer da criação do modelo;

## Aditivo1:
Objetivo
: Separar o arquivo de lemantizer da criação do modelo;

Resumo
: O objetivo é separar o código que realiza a lemantização do código que gera o modelo afim de permitir que sejam inicializados modelos distintos sem necessidade de alteração no lemantizador.


