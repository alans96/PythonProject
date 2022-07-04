"""
Testar qual os parametros melhores para criar uma rede
ESTE TESTE PODE DEMORAR HORAS PARA TERMINAR

@author: @alan96_s
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('breast_entradas.csv')
classe = pd.read_csv('breast_saidas.csv')

#Criar rede neural 
def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units=16, activation = activation, 
                        kernel_initializer= kernel_initializer, input_dim = 30)) #1 camada oculta
    
    classificador.add(Dropout(0.2)) #Das 30 entradas zera 20%  para evitar o overfitting
    
    classificador.add(Dense(units=neurons, activation=activation, 
                        kernel_initializer= kernel_initializer))#2 camada oculta
    
    classificador.add(Dropout(0.2))#Das 16 entradas zera 20%  para evitar o overfitting
    
    
    #camada de saída
    classificador.add(Dense(units=1, activation= 'sigmoid')) 


#Classificador padão
    classificador.compile(optimizer=optimizer, loss =loss,
                      metrics=['binary_accuracy']) #compilar a rede

    return classificador

classificador = KerasClassifier(build_fn= criarRede)
#parametros que vai testar
parametros= {'batch_size': [10,30],
             'epochs': [50,100],
             'optimizer': ['adam', 'sgd'],
             'loos': ['binary_crossentropy', 'hinge'],
             'kernel_initializer': ['random_uniform', 'normal'],
             'activation': ['relu', 'tanh'],
             'neurons': [16,8]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid= parametros,
                           scoring = 'accuracy',
                           cv=5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
r= # ERRO PARA NÃO RODAR POR ACIDENTE