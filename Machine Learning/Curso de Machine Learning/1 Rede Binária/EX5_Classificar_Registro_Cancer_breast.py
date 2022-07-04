"""
Criar um registro, depois de fazer o Turing, descobre os melhores valores
agora criando ar a rede 

@author: @alan96_s
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('breast_entradas.csv')
classe = pd.read_csv('breast_saidas.csv')


classificador = Sequential()
classificador.add(Dense(units=8, activation = 'relu', 
                        kernel_initializer= 'normal', input_dim = 30)) #1 camada oculta
    
classificador.add(Dropout(0.2)) #Das 30 entradas zera 20%  para evitar o overfitting
    
classificador.add(Dense(units=8, activation='relu', 
                        kernel_initializer= 'normal'))#2 camada oculta
    
classificador.add(Dropout(0.2))#Das 16 entradas zera 20%  para evitar o overfitting
    
    
#camada de saída
classificador.add(Dense(units=1, activation= 'sigmoid')) 

#Classificador padão
classificador.compile(optimizer='adam', loss ='binary_crossentropy',
                      metrics=['binary_accuracy']) #compilar a rede

classificador.fit(previsores, classe, batch_size=10, epochs=50)


novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5) # intes do numero falar true ou False a depender do 0.5











