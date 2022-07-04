"""
Para gravar a rede neural em 2 arquivos

@author: @alan96_s
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

classificador_json = classificador.to_json()# cria as configurações da rede

with open ('classificador_breast.json', 'w') as json_file: #para salvar em disco
    json_file.wrile(classificador_json) 
    
classificador.save_weights('classificador_breast.h5') #Para salvar os pesos / instalar h5.py