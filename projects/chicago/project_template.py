import csv

print("Reading the document...")
with open("chicago.csv", "r") as file_read:
    reader = csv.reader(file_read)
    data_list = list(reader)
print("Ok!")

# Vamos ver quantas linhas o documento possui
print("Number of rows:")
print(len(data_list))

# Vamos ver a primeira linha do documento
print("Row 0: ")
print(data_list[0])
# É o header. É pelo header que identificamos as variáveis (colunas)

# Vamos ver qual a segunda linha de data_list.
# Essa linha já deve conter alguns dados
print("Row 1: ")
print(data_list[1])

input("Press Enter to continue...")
# TASK 1
# TODO: Printe as primeiras 20 linhas usando um loop para identificar os dados.
print("\n\nTASK 1: Printing the first 20 samples")


# Vamos printar o enumerate do header, pois o removeremos do dataset logo após
print(list(enumerate(data_list[0])))
# Removendo o header
data_list = data_list[1:]

# Podemos acessar as variáveis (features, colunas) pelo seu index
# sample[6] ou sample[-2] pra acessar o gênero de um caso


input("Press Enter to continue...")
# TASK 2
# TODO: Printe o gênero dos primeiros 20 casos (rows, samples, cases)
print("\nTASK 2: Printing the genders of the first 20 samples")

# Legal, nós podemos acessar alguns casos iterando com um for loop e suas features por index.
# Entretanto, o ideal seria ter as colunas em listas. Exemplo: Lista com os gêneros de todos os casos

input("Press Enter to continue...")
# TASK 3
# TODO: Crie uma função para adicionar as features de uma lista em outra lista da mesma ordem
def column_to_list(data, index):
    column_list = []
    # Dica: você pode usar um loop for para iterar sobre os casos, pegar a feature por index e dar um append na lista
    return column_list
    


    
# Vamos checar com os gêneros para ver se está funcionando. Considere apenas os 20 primeiros casos
print("\nTASK 3: Printing the list of genders of the first 20 samples")
print(column_to_list(data_list, -2)[:20])

# ------------ NÃO MUDE NADA DAQUI!!!!!!!! ------------
assert type(column_to_list(data_list, -2)) is list, "TASK 3: Wrong type returned. It should return a list."
assert len(column_to_list(data_list, -2)) == 1551505, "TASK 3: Wrong lenght returned."
assert column_to_list(data_list, -2)[0] == "" and column_to_list(data_list, -2)[1] == "Male", "TASK 3: The list doesn't match."
# -----------------------------------------------------

input("Press Enter to continue...")
# Beleza, agora que sabemos como contar as features, vamos contar quantos Homens e Mulheres (Males e Females) têm no conjunto de dados
# TASK 4
# TODO: Conte cada gênero. Não use nenhuma função pre-built
male = 0
female = 0

print("\nTASK 4: Printing how many males and females we found")
print("Male: {}\nFemale: {}".format(male, female))


# ------------ NÃO MUDE NADA DAQUI!!!!!!!! ------------
assert male == 935854 and female == 298784, "TASK 4: Count doesn't match."
# -----------------------------------------------------

input("Press Enter to continue...")
# Por que não criamos uma função para fazer esse cálculo de Males e Females?
# TASK 5
# TODO: Crie uma função que conte os gêneros e retorna uma lista
# Deve retornar uma lista com [count_male, count_female], exemplo: [10, 15] para 10 homens e 15 mulheres
def count_gender(data_list):
    male = 0
    female = 0
    return [male, female]
    

print("\nTASK 5: Printing result of count_gender")
print(count_gender(data_list))

# ------------ NÃO MUDE NADA DAQUI!!!!!!!! ------------
assert type(count_gender(data_list)) is list, "TASK 5: Wrong type returned. It should return a list."
assert len(count_gender(data_list)) == 2, "TASK 5: Wrong lenght returned."
assert count_gender(data_list)[0] == 935854 and count_gender(data_list)[1] == 298784, "TASK 5: Returning wrong result!"
# -----------------------------------------------------

input("Press Enter to continue...")
# Será qual o gênero que mais usa o sistema de compartilhamento de bicicles?
# TASK 6
# TODO: Crie uma função que retorne o gênero que mais usa e retorne o gênero como string
# Deve ser uma função lógica e os retornos possíveis são: "Male", "Female" ou "Equal"
def most_popular_gender(data_list):
    answer = ""
    return answer
    
print("\nTASK 6: Which one is the most popular gender?")
print("Most popular gender is: {}".format(most_popular_gender(data_list)))

# ------------ NÃO MUDE NADA DAQUI!!!!!!!! ------------
assert type(most_popular_gender(data_list)) is str, "TASK 6: Wrong type returned. It should return a string."
assert most_popular_gender(data_list) == "Male", "TASK 6: Returning wrong result!"
# -----------------------------------------------------

# Se até então está tudo rodando bonitinho, chequem este gráfico
# Depois aprenderemos a como plotar lindos gráficos :3
#quantity = count_gender(data_list)
#y_pos = list(range(len(types)))
#plt.bar(y_pos, quantity)
#plt.ylabel('Quantity')
#plt.xlabel('Gender')
#types = ["Male", "Female"]
#plt.xticks(y_pos, types)
#plt.title('Quantity by Gender')
#plt.show(block=True)

input("Press Enter to continue...")
# TASK 7
# TODO: Responda a seguinte pergunta:
male, female = count_gender(data_list)
print("\nTASK 7: Why the following condition is False?")
print("male + female == len(data_list):", male + female == len(data_list))
answer = "Type your answer here."
print("Answer:", answer)

# ------------ NÃO MUDE NADA DAQUI!!!!!!!! ------------
assert answer != "Type your answer here.", "TASK 7: Write your own answer!"
# -----------------------------------------------------

input("Press Enter to continue...")
# Vamos trabalhar com a feature trip_duration agora.
# TASK 8
# TODO: Encontre o mínimo, máximo, média e a mediana de trip_duration
# Não use funções built-in. como min(), max(), mean(), median()
# A única função built-in que voce pode usar é a sort() ou sorted()
# Dica, é bom ordernar a lista de trip duration para responder todas as perguntas
trip_duration_list = column_to_list(data_list, 2)
min_trip = 0.
max_trip = 0.
mean_trip = 0.
median_trip = 0.

print("\nTASK 8: Printing the min, max, mean and median")
print("Min: ", min_trip, "Max: ", max_trip, "Mean: ", mean_trip, "Median: ", median_trip)

# ------------ NÃO MUDE NADA DAQUI!!!!!!!! ------------
assert round(min_trip) == 60, "TASK 8: min_trip with wrong result!"
assert round(max_trip) == 86338, "TASK 8: max_trip with wrong result!"
assert round(mean_trip) == 940, "TASK 8: mean_trip with wrong result!"
assert round(median_trip) == 670, "TASK 8: median_trip with wrong result!"
# -----------------------------------------------------


input("Press Enter to continue...")
# TASK 9
# TODO: Verifique quantas start_stations nós temos
# Dica: O tipo de dado set remover duplicatas (valores repetidos)
user_types = 0

print("\nTASK 9: Printing start stations:")
print(len(user_types))
print(user_types)

# ------------ NÃO MUDE NADA DAQUI!!!!!!!! ------------
assert len(user_types) == 582, "TASK 9: Wrong len of start stations."
# -----------------------------------------------------

input("Press Enter to continue...")
# TASK 10 - DESAFIO MORRRR! <3
# TODO: Crie uma função para contar tipos de usuário, sem dizer exatamente quais são os tipos de usuário
# assim, poderemos usar essa função para qualquer tipo de dado
print("Will you face it?")
answer = "no"

def count_items(column_list):
    item_types = []
    count_items = []
    return item_types, count_items


if answer == "yes":
    # ------------ DO NOT CHANGE ANY CODE HERE ------------
    column_list = column_to_list(data_list, -2)
    types, counts = count_items(column_list)
    print("\nTASK 10: Printing results for count_items()")
    print("Types:", types, "Counts:", counts)
    assert len(types) == 3, "TASK 10: There are 3 types of gender!"
    assert sum(counts) == 1551505, "TASK 10: Returning wrong result!"
    # -----------------------------------------------------