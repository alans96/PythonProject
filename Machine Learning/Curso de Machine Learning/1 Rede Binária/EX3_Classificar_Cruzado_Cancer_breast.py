"""
Uma Classificação  cruzada binária se possui ou não um tumor

@author: @alan96_s
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('breast_entradas.csv')
classe = pd.read_csv('breast_saidas.csv')

#Criar rede neural 
def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units=16, activation='relu', 
                        kernel_initializer= 'random_uniform', input_dim = 30)) #1 camada oculta
    
    classificador.add(Dropout(0.2)) #Das 30 entradas zera 20%  para evitar o overfitting
    
    classificador.add(Dense(units=16, activation='relu', 
                        kernel_initializer= 'random_uniform'))#2 camada oculta
    
    classificador.add(Dropout(0.2))#Das 16 entradas zera 20%  para evitar o overfitting
    

    classificador.add(Dense(units=1, activation= 'sigmoid')) #camada de saída


#Classificador padão
    classificador.compile(optimizer='adam', loss ='binary_crossentropy',
                      metrics=['binary_accuracy']) #compilar a rede

    return classificador


classificador = KerasClassifier(build_fn=criarRede, epochs= 100, batch_size= 10)

resultados = cross_val_score(estimator = classificador, X = previsores,y = classe, cv =5, scoring = 'accuracy')

media = resultados.mean()
desvio= resultados.std()




















