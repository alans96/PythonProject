"""
fazer uma previsão de um valor para a ação

@author: alan96_s
"""
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense, Dropout, LSTM # atualizado: tensorflow==2.0.0-beta1
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importação do arquivo
base = pd.read_csv('petr4-treinamento.csv')
base = base.dropna() # apagar valores vazios(ana) 
#todas as colunas de 1 até a 2(no final vai ser somente a 1)
base_treinamento = base.iloc[:, 1:2].values

#normalizar os valores de 0 a 1
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = [] #valores de base do preço
preco_real = [] #valor previsto
#usar 90 valores para usar como base do preço
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])#linha, coluna
    preco_real.append(base_treinamento_normalizada[i, 0])
#adaptando a tabela
previsores, preco_real = np.array(previsores), np.array(preco_real)
#formato para o keras
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))#1=um indicador

#Criar a rede
regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

#2 camada
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

#3 camada
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

#4Camada
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

#Resposta final
regressor.add(Dense(units = 1, activation = 'linear'))

#Compilação
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error'])
#Treinar
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)



base_teste = pd.read_csv('petr4-teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
#adaptar os valores e mostrar os 90 valores anteriores de cada dia
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

#90 + 22 = 122
X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()
   
#Criar gráfico 
plt.plot(preco_real_teste, color = 'red', label = 'PreÃ§o real')
plt.plot(previsoes, color = 'blue', label = 'PrevisÃµes')
plt.title('PrevisÃ£o preÃ§o das aÃ§Ãµes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()