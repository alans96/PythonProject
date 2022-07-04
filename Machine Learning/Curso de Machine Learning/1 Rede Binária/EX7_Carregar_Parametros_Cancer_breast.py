"""
Para gravar a rede neural em um arquivo

@author: @alan96_s
"""

import numpy as np
import pandas as pd
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutura_rede = arquivo.read() # copiou os dados do arquivo para a estrutura_rede
arquivo.close()

classificador = model_from_json(estrutura_rede) # carrega a estrutura da rede 
classificador.load_weights('classificador_breast.h5') # Carrega os pessos

#adicionar um registro e compara-lo
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5) # intes do numero falar true ou False a depender do 0.5

# Ver o resultado em toda a base de dados

previsores = pd.read_csv('breast_entradas.csv')
classe = pd.read_csv('breast_saidas.csv')

classificador.compile(optimizer='adam', loss ='binary_crossentropy',
                      metrics=['binary_accuracy'])
resultado  = classificador.evaluate(previsores, classe)






