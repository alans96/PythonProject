"""
Testar qual os parametros melhores para criar uma rede


@author: @alan96_s
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix


base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4].values # iloc pega parte do csv e pegar todas as linhas e 3 colunas
classe = base.iloc[:,4].values # pega a coluna 4


#Fazer a saida que é stq passar para numero
labeencorder = LabelEncoder()
classe = labeencorder.fit_transform(classe) # a classe recebe ela transformada
classe_final = np_utils.to_categorical(classe)
#iris setosa 1 0 0
#iris virgica 0 1 0
#iris versicolor 0 0 1


previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores,
                                                                                              classe_final, test_size=0.25)
classificador = Sequential()

#primeira camada
classificador.add(Dense(units= 4, activation=('relu'),input_dim= 4)) #input dim 4 atributos de entrada

#Segunda camada
classificador.add(Dense(units=4, activation='relu'))

#camada de saída
classificador.add(Dense(units=3, activation='softmax' )) #softmax para saida maiores ou igual a 3 classes


classificador.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics= ['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs= 500)

resultado = classificador.evaluate(previsores_teste,classe_teste)

#Criar a matriz de confusão (comparação)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes >0.5) # passar para true e false

#Retornar o indice de cada linha, exemplo classificou como setosa, retorna o indice 1
classe_teste2 = [np.argmax(t) for t in classe_teste] 
previsoes2 = [np.argmax(t) for t in previsoes]

#Cria a matriz de confusão
matriz = confusion_matrix(previsoes2, classe_teste2)

