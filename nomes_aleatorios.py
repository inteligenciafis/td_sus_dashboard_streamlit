from br_nome_gen import pessoa_random
import pandas as pd
pessoas = []
for i in range(100):
    pessoas.append(pessoa_random())
pessoas = pd.DataFrame(pessoas)
homens = pessoas[pessoas.masc == True]
mulheres = pessoas[pessoas.masc == False]

#mulheres.to_csv("nome_mulheres.csv")
#homens.to_csv("nome_homens.csv")

pacientes = pd.read_csv("pacientes15.csv")
# mulheres = pd.read_csv("nome_mulheres.csv")
# homens = pd.read_csv("nome_homens.csv")

nomes = []
for i in range(len(pacientes)): 
    if pacientes.SEXO[i] == 3:
        nomes.append(mulheres.nome.iloc[i])
    else:
        nomes.append(homens.nome.iloc[i])

pacientes["nome"] = nomes

pacientes.to_csv("pacientes15cnome.csv")