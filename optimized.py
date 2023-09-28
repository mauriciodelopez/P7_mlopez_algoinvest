import pandas as pd
import time


start_time = time.time()


# Definir variables
budget = 500.0  # Change the budget for a float // 
best_profit = 0.0
best_combinations = []
#plus de all_combinaisons

# Ouvrir dataset
url = 'dataset/dataset2_Python.csv'

df = pd.read_csv(url, sep=";")          #O(n) omplexité temporelle, où n est le nombre de lignes dans le fichier.

#création de colonnes pour le rendre générique 
first_column = df.columns[1]
second_column = df.columns[2]

df = df[df[first_column]>0]   #df 
#actionSG = df.loc[740, first_column, second_column ]

#exploration de data
print("colmuns of dataframe", df.columns)
print("shape of dataframe ", df.shape)
print("info of dataframe", df.info)
print("head of dataframe", df.head)
print("describe of dataframe", df.describe)
# Contar valores nulos en todo el DataFrame
print("count null values of dataframe", df.isnull().sum())
print("duplicated of dataframe", df.duplicated)

# Register: obtenir le dataframe et le convertir en un dictionnaire
register = df.to_dict("records")

#création des deux lists profits & costs pour un acces plus rapid, on supprimant le "for" eviter que se repete plusieurs fois
profits = [float(action[second_column]) for action in register]  # Convertir a point float
costs = [float(action[first_column]) for action in register]     # Convertir a point float  //O(n)complexité pour les deux listes car elles itèrent sur toutes les lignes du DataFrame

# Création d'une matriz afin de suivre les meilleurs combinaisons
n = len(register) #n es la longitud de todos los registro osea el tamano
dp = [[0.0] * int(budget) for _ in range(n)]  # Cambiamos a valores de punto flotante y entero //complexité de O(n * budget), car il y a une double boucle imbriquée qui parcourt n et budget
#dp c'est une matriz et sert a faire le suivie des meilleurs combos possibles
#Le 0 est le point de départ de la matriz multiplier par le budget puis le cicle for pour les files et trouver les meilleurs combinaisons

#Il parcourt deux boucles, une pour i comprise entre 1 et n-1 et une autre pour j comprise entre 0 et int(budget) - 1. 
# Ces boucles sont utilisées pour considérer chaque élément (indexé par i) et chaque montant budgétaire possible (indexé par j).
for i in range(1, n):
    for j in range(int(budget)): #A l'intérieur de la boucle, il vérifie si le coût de l'élément actuel (coûts[i - 1]) est inférieur ou égal au budget actuel (j). 
          if costs[i - 1] <= j: #Si tel est le cas, il calcule le profit maximum pouvant être obtenu avec et sans sélection de l'élément actuel.
              dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - (int(costs[i - 1]))] + profits[i - 1])#Cette ligne met à jour la valeur dans le tableau dp pour l'élément et le budget actuels en prenant le maximum de deux possibilités :
          else:
              dp[i][j] = dp[i - 1][j]# si depasse j reste aven l'element actuel;représente le profit maximum obtenu sans sélectionner l'élément actuel.
    



# Reconstruir las meilleurs combinaisons
aux=0 #aux est initialisé à 0  
j = int(budget-1)#j est initialisé à int(budget) - 1. aux sera utilisé pour suivre le coût cumulé des éléments sélectionnés, et j représente le budget restant.
for i in range(n-1, 0, -1): #Il démarre une boucle qui itère dans l'ordre inverse de n-1 jusqu'à 1. Cette boucle est utilisée pour revenir du dernier élément au premier élément de la table de programmation dynamique dp.
    if dp[i][j] != dp[i - 1][j] and (aux+register[i-1][first_column]<=budget): #A l'intérieur de la boucle, il vérifie si la valeur de dp[i][j] n'est pas égale à dp[i - 1][j]. Cette condition implique que l'élément actuel i-1 a été sélectionné dans la solution optimale (car le profit maximum a changé).
        aux+=register[i-1][first_column] #Il vérifie également si l'ajout du coût de l'élément actuel (register[i-1][first_column]) à aux (coût cumulé des éléments sélectionnés) permettrait toujours de maintenir le coût total dans le budget. Si tel est le cas, cela signifie que l’élément actuel peut être inclus dans la solution.
        best_combinations.append(register[i - 1]) #Si les deux conditions sont remplies, il met à jour aux en y ajoutant le coût de l'élément actuel (aux += register[i-1][first_column]), ajoute l'élément correspondant à la liste best_combinations (best_combinations.append(register[i - 1])), et décrémente j de int(costs[i - 1]), ce qui représente une réduction du budget disponible.
        j -= int(costs[i - 1])#La boucle continue jusqu'à avoir parcouru tous les éléments du dernier au premier en respectant la contrainte budgétaire.

#linea 53 valida si entra en el budget sinon pasa a otra accion 

best_combinations_df = pd.DataFrame(best_combinations)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(best_combinations_df)



total_profit = best_combinations_df[second_column].sum()
print("\nTotal profits:", total_profit)

total_cost = best_combinations_df[first_column].sum()
print("Total cost:", total_cost)
print('Nombre Total de combinations: ', len(best_combinations))

end = time.time()
print("\n execution time", end - start_time)




#este hay que recopiarlo si besoin este tiene las ultimas explicaciones de sofia 


"""# Importa la biblioteca pandas para trabajar con datos tabulares y time para medir el tiempo de ejecución.
import pandas as pd
import time

# Registra el tiempo de inicio de la ejecución.
start_time = time.time()

# Define la variable budget (presupuesto) con un valor de punto flotante de 500.0.
budget = 500.0

# Inicializa las variables best_profit (mejor beneficio) y best_combinations (mejores combinaciones) como vacías.
best_profit = 0.0
best_combinations = []

# Especifica la URL del archivo de datos CSV.
url = '/content/dataset2_Python+P7.csv'

# Lee el conjunto de datos desde la URL utilizando pandas, especificando que el separador es ";".
df = pd.read_csv(url, sep=";")

# Extrae los nombres de las columnas relevantes en el conjunto de datos.
first_column = df.columns[1]
second_column = df.columns[2]

# Filtra las filas del conjunto de datos donde el valor en la primera columna sea mayor que 0.
df = df[df[first_column] > 0]

# Convierte el DataFrame df en un diccionario llamado register.
register = df.to_dict("records")

# Crea dos listas, profits y costs, que contienen los valores de beneficio y costo, respectivamente, desde el diccionario register.
profits = [float(action[second_column]) for action in register]
costs = [float(action[first_column]) for action in register]

# Inicializa una matriz dp con dimensiones n x budget para realizar un seguimiento de los resultados óptimos de subproblemas.
n = len(register)
dp = [[0.0] * int(budget) for _ in range(n)]

# Resuelve el problema de la mochila utilizando programación dinámica.
for i in range(1, n):
    for j in range(int(budget)):
        # Verifica si el costo del elemento actual es menor o igual al presupuesto actual.
        if costs[i - 1] <= j:
            # Calcula el máximo entre dos opciones: 
            # (1) no incluir el elemento actual (tomando el valor de la fila anterior dp[i-1][j])
            # (2) incluir el elemento actual, restando su costo y sumando su beneficio al valor de la fila anterior dp[i-1][j - abs(int(costs[i - 1]))] + profits[i - 1])
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - abs(int(costs[i - 1]))] + profits[i - 1])
        else:
            # Si el costo del elemento actual es mayor que el presupuesto actual, simplemente copia el valor de la fila anterior.
            dp[i][j] = dp[i - 1][j]

# Reconstruye las mejores combinaciones.
aux = 0
j = int(budget - 1)
for i in range(n - 1, 0, -1):
    # Comprueba si el valor en la matriz dp cambió en comparación con el valor de la fila anterior en la misma columna.
    if dp[i][j] != dp[i - 1][j] and (aux + register[i - 1][first_column] <= budget):
        # Si el valor cambió y el costo del elemento actual no excede el presupuesto restante, agrega el elemento a las mejores combinaciones.
        aux += register[i - 1][first_column]
        best_combinations.append(register[i - 1])
        j -= int(costs[i - 1])

# Crea un DataFrame best_combinations_df a partir de las mejores combinaciones.
best_combinations_df = pd.DataFrame(best_combinations)

# Configura opciones de visualización de pandas para mostrar todas las filas y columnas en la salida.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Imprime las mejores combinaciones.
print(best_combinations_df)

# Calcula el beneficio total y el costo total de las mejores combinaciones.
total_profit = best_combinations_df[second_column].sum()
total_cost = best_combinations_df[first_column].sum()

# Imprime el beneficio y costo totales.
print("\nTotal de beneficios:", total_profit)
print("Total de costos:", total_cost)

# Registra el tiempo de finalización de la ejecución y muestra el tiempo total de ejecución.
end = time.time()
print("\nTiempo de ejecución:", end - start_time)"""