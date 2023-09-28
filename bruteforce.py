import itertools
import pandas as pd
import time
from tqdm import tqdm

start_time = time.time()

# Define variables
budget = 500
best_profit = 0
all_combinations = []
best_combinations = []
best_total_cost = 0

# Open dataset
url = 'dataset/fbrute_actions.csv'
df = pd.read_csv(url, sep=";")
register = df.to_dict("records")
num_records = len(register)

# Define a list to store dictionaries representing the best combinations
best_combinations_dict = []

# Cycle to generate all possible combinations of indices of records
for i in tqdm(range(1, num_records + 1)):
    for combination_indices in itertools.combinations(range(num_records), i):
        combination = [register[index] for index in combination_indices]
        all_combinations.append(combination)
        total_cost = sum(int(action["Coût par action (en euros)"]) for action in combination)
        total_profit = sum(int(action["Bénéfice (après 2 ans)"]) for action in combination)
        
        if total_cost <= budget:
            if total_profit > best_profit:
                best_combinations_dict = []
                best_profit = total_profit
                best_total_cost = total_cost  # Update the best_total_cost
            if total_profit >= best_profit:
                best_combinations_dict.append({"Combination": combination, "Total_Cost": total_cost, "Total_Profit": total_profit})

# Create a list of dictionaries for flattened combinations
flattened_combinations = []
for entry in best_combinations_dict:
    combination = entry["Combination"]
    total_cost = entry["Total_Cost"]
    total_profit = entry["Total_Profit"]
    for action in combination:
        flattened_combinations.append({"Action": action["Actions #"], "Coût par action (en euros)": 
            action["Coût par action (en euros)"], "Bénéfice (après 2 ans)": action["Bénéfice (après 2 ans)"], 
            "Total_Cost": total_cost, "Total_Profit": total_profit}
                                      )

# Convert the list of dictionaries into a DataFrame
best_combinations_df = pd.DataFrame(flattened_combinations)

print(best_combinations_df)
print("Total number of combinations evaluated:", len(all_combinations))
print("Total cost of the best combination:", best_total_cost)  # Corrected print statement

end = time.time()
print("Execution time :", end - start_time)

