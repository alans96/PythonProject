# -*- coding: utf-8 -*-
"""
Uma Classificação binária se possui ou não um tumor

@author: @alan96_s
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf


previsores = pd.read_csv('breast_entradas.csv')
classe = pd.read_csv('breast_saidas.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25) 
# Test zise é usar 25% so arquivo para testar o funcionamento

classificador = Sequential()
classificador.add(Dense(units=16, activation='relu', 
                        kernel_initializer= 'random_uniform', input_dim = 30)) #1 camada oculta
# unitis é os primeiros neurinios ((entradas + saidas)/2)
#imput 30 pq têm 30 entradas


classificador.add(Dense(units=16, activation='relu', 
                        kernel_initializer= 'random_uniform'))#2 camada oculta

classificador.add(Dense(units=1, activation= 'sigmoid')) #camada de saída

#otimizar os parametros do Adam

'''otimizador = tf.keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)

classificador.compile(optimizer=otimizador, loss ='binary_crossentropy',
                      metrics=['binary_accuracy']) #compilar a rede'''

#Classificador padão
classificador.compile(optimizer='adam', loss ='binary_crossentropy',
                      metrics=['binary_accuracy']) #compilar a rede

# Teinar a rede
classificador.fit(previsores_treinamento,classe_treinamento, batch_size=10,
                  epochs=100) 

previsoes = classificador.predict(previsores_teste) #previsão
previsoes = (previsoes >0.5) # a função sigmoid diz que para valor acima de 0.5 é true

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)# comparação

resultado = classificador.evaluate(previsores_teste, classe_teste)#primeiro valor é o erro, segundo a precisão