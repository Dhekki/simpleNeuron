import pandas as pd
import math
import numpy as np

url = "https://raw.githubusercontent.com/nobertomaciel/AI-UNIFACS/refs/heads/main/dados.csv"
df = pd.read_csv(url, header=None)
dataset_np = df.to_numpy()
print(len(df))
u = 0
n = 0.1
pesos = np.array([])
row = np.array([])
np.random.seed(40)
saidasReais = dataset_np[:, 4]
saidasTreino = []
saidasTeste = []
errosTreino = []
errosTeste = []

def errosConvertidos():
  global errosTreino, errosTeste
  errosTreino = [float(erro) for erro in errosTreino]
  errosTeste = [float(erro) for erro in errosTeste]

def taxaAcerto():
  global errosTeste
  acertos = 0
  for erro in errosTeste:
    if erro == 0:
      acertos += 1
  return (acertos / len(errosTeste)) * 100

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
  global pesos, u, entradas, dataset_np, row, e, n
  epoca = 1
  e = float('inf')
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

def percorrerDados(indiceInicial, subtracaoFinal):
  global pesos, u, entradas, dataset_np, row
  eTreino = True
  for i in range(indiceInicial, len(dataset_np) - subtracaoFinal):
    row = dataset_np[i]
    entradas = row[:4] #2
    rodada = i + 1
    print("Rodada: " + str(rodada) + "\n")

    if indiceInicial == 20:
      eTreino = False
    if eTreino:
      treinamento()
    else:
      u = 0
      g = degrauBipolar()
      saidasTeste.append(g)
      e = row[4] - g
      errosTeste.append(e)


percorrerDados(0, 10) #1º ao 20º
print("\n-----------------------------------------\n")
percorrerDados (20, 0) #21º ao 30º

errosConvertidos()
acuracia = taxaAcerto()

print("Saídas esperadas: " + str(saidasReais))
print("Saídas obtidas (treino): " + str(saidasTreino))
print("Saídas obtidas (teste): " + str(saidasTeste))
print("Erros (treino): " + str(errosTreino))
print("Erros (teste): " + str(errosTeste))
print("Acurácia: " + str(acuracia) + "%")
