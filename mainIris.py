import pandas as pd
import math
import numpy as np
from sklearn.datasets import load_iris

dadosIris = load_iris()
dadosSP = dadosIris['data'][:, :]
saidas = np.zeros(dadosSP.shape[0])
saidas[:50] = 1
saidas[50:] = -1
dadosSP = np.column_stack((dadosSP, saidas))
# print(dadosSP)

setosaTreino = dadosSP[:25, :]
versicolorTreino = dadosSP[50:75, :]
virginicaTreino = dadosSP[100:125, :]

setosaTeste = dadosSP[:25, :]
versicolorTeste = dadosSP[50:75, :]
virginicaTeste = dadosSP[100:125, :]

dadosTreino = np.concatenate((setosaTreino, versicolorTreino, virginicaTreino))
dadosTeste = np.concatenate((setosaTeste, versicolorTeste, virginicaTeste))

e = float('inf')
epoca = 1
n = 0.1
pesos = np.array([])
row = np.array([])
# dataset_np = np.array(dadosSP)
datasetTreino = np.array(dadosTreino)
datasetTeste = np.array(dadosTeste)
np.random.seed(40)
eTreino = True
# saidasReais = dataset_np[:, 4]
saidasReaisTreino = datasetTreino[:, 4]
saidasReaisTeste = datasetTeste[:, 4]
saidasTreino = []
saidasTeste = []
errosTreino = []
errosTeste = []

def errosConvertidos():
  global errosTreino, errosTeste
  errosTreino = [float(erro) for erro in errosTreino]
  errosTeste = [float(erro) for erro in errosTeste]

def taxaAcerto(erros):
  erros
  acertos = 0
  for erro in erros:
    if erro == 0:
      acertos += 1
  return (acertos / len(erros)) * 100

def degrauBipolar():
  global u, entradas, pesos
  for j, entrada in enumerate(entradas):
      u += entrada * pesos[j] #3
  if u > 0: #4
    g = 1
  elif u == 0:
    g = 0
  else:
    g = -1
  print("\nAtivação: u: " + str(u) + " g: " + str(g) + "\n")
  return g

def treinamento():
  global pesos, u, entradas, row, e, epoca, n, e
  if pesos.size == 0:
    pesos = np.random.uniform(-0.5, 0.5, len(entradas)) #1

  while (e != 0 and epoca < 10):
    u = 0
    g = degrauBipolar()

    print("Época " + str(epoca))
    for k, entrada in enumerate(entradas):
      nEntrada = k + 1
      print("W" + str(nEntrada) + ": " + str(pesos[k]))
      pesos[k] = pesos[k] + n * (row[4] - g) * entrada # 5
      print("W" + str(nEntrada) + "n: " + str(pesos[k]) + " | n: " + str(n) + " | Yreal: " + str(row[4]) +
            " | Ypred: " + str(g) + " | X" + str(nEntrada) + ": " + str(entrada))

    e = row[4] - g #6
    print("Erro: " + str(e))
    print("")
    if e == 0:
      saidasTreino.append(g)
      errosTreino.append(e)
    # print("Erro: " + str(e))
    # print("")
    epoca += 1
  print("")

def percorrerDados(dataset_np):
  global eTreino, pesos, u, entradas, row, e, epoca
  for i in range(len(dataset_np)):
    row = dataset_np[i]
    entradas = row[:4] #2
    e = float('inf')
    epoca = 1
    rodada = i + 1
    print("Rodada: " + str(rodada) + "\n")

    # if indiceInicial == 20:
    #   eTreino = False
    if eTreino:
      treinamento()
    else:
      u = 0
      g = degrauBipolar()
      saidasTeste.append(g)
      e = row[4] - g
      errosTeste.append(e)
  eTreino = False


percorrerDados(datasetTreino)
print("\n-----------------------------------------\n")
percorrerDados (datasetTeste)

errosConvertidos()
acuraciaTreino = taxaAcerto(errosTreino)
acuraciaTeste = taxaAcerto(errosTeste)

print("Saídas esperadas (treino): " + str(saidasReaisTreino))
print("Saídas obtidas (treino): " + str(saidasTreino))
print("Saídas esperadas (teste): " + str(saidasReaisTeste))
print("Saídas obtidas (teste): " + str(saidasTeste))
print("Erros (treino): " + str(errosTreino))
print("Erros (teste): " + str(errosTeste))
print("Acurácia (treino): " + str(acuraciaTreino) + "%")
print("Acurácia (teste): " + str(acuraciaTeste) + "%")
