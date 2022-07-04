"""
Informal qual o valor do carro

@author: alan_96s
"""

import pandas as pd
import tensorflow as tf # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.models import Sequential # atualizado: tensorflow==2.0.0-beta1
from tensorflow.keras.layers import Dense # atualizado: tensorflow==2.0.0-beta1

# Carregar a base
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

# Excluir a coluna de dados
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

#Escluir colunas
base['name'].value_counts()
base = base.drop('name', axis = 1)
base['seller'].value_counts()
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

# Escluir linhas com preço errado
i1 = base.loc[base.price <= 10]
base.price.mean()
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

#Mostrar qual os valores que mais aparece
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein

#Criar um dic com os items que mais aparece
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
#Colocar na base os valores onde não tem valor
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 0 0 0 0
# 2 0 1 0
# 3 0 0 1

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough') #atualizado
previsores = onehotencoder.fit_transform(previsores).toarray() #atualizado

regressor = Sequential([ # atualizado: tensorflow==2.0.0-beta1
        tf.keras.layers.Dense(units=158, activation = 'relu', input_dim=316),
        tf.keras.layers.Dense(units=158, activation = 'relu'),
        tf.keras.layers.Dense(units=1, activation = 'linear')])

regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)
preco_real.mean()
previsoes.mean()




















