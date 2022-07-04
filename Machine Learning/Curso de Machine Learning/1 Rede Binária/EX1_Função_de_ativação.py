"""
Created on Thu Mar 24 09:50:50 2022

@author: @alan96_s
"""

import numpy as np


def stepFuntion(soma):
    '''Diz se vai ser 0 ou 1'''
    if (soma>=1):
        return 1
    return 0


def sigmoidFunction(soma):
    '''Diz valores de 0 a 1'''
    return 1/ (1 + np.exp(-soma))


def tahnFunction(soma):
    '''Diz valores de -1 a 1'''
    return (np.exp(soma)- np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))


def reluFunction(soma):
    '''Diz valores >= 0'''
    if soma >= 0:
        return soma
    return 0


def linearFunction(soma):
    '''retorna o valor'''
    return soma


def softmaxFunction(x):
    '''Retorna probabilidades de acordo com a saída '''
    ex = np.exp(x)
    return ex/ ex.sum()
    

teste = stepFuntion(30)
teste2 = sigmoidFunction(0.358)       
teste3 = tahnFunction(-0.358)
teste4 = reluFunction(-10)
teste5 = linearFunction(12)
valores = [5,2,3]
print(softmaxFunction(valores))
