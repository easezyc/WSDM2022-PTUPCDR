import pandas as pd
from scipy import stats
import random

mine = pd.read_csv("/Users/dipakmeher/Documents/GMU/GMU_Semester/Summer-2023/GRA-Dr.Rosenblum/GRA_Work/stats/3_7/tgt_games_src_electronics/3_7_games_electronics_stats.csv")
# print(mine)

ptu = pd.read_csv("/Users/dipakmeher/Documents/GMU/GMU_Semester/Summer-2023/GRA-Dr.Rosenblum/GRA_Work/Research-PTUPCDR/rev_rmse_3_7_T4_GMF.csv")
# print(ptu)

# my_rmse = mine['rmse']
# ptu_rmse = ptu['RMSE']

# my_mae = mine['mae']
# ptu_mae = ptu['MAE']

# # print(my_rmse)
# # print(ptu_rmse)

# mae_t_stat, mae_p_value = stats.ttest_rel(my_mae, ptu_mae)

# print("T-stat MAE: " + str(mae_t_stat))
# print("P-value MAE: " + str(mae_p_value))

# rmse_t_stat, rmse_p_value = stats.ttest_rel(my_rmse, ptu_rmse)

# print("T-stat RMSE: " + str(rmse_t_stat))
# print("P-value RMSE: " + str(rmse_p_value))

# Select random 100 values from both datasets
random_indices = random.sample(range(len(mine)), 100)
mine_sample = mine.iloc[random_indices]
ptu_sample = ptu.iloc[random_indices]

my_rmse = mine_sample['rmse']
ptu_rmse = ptu_sample['RMSE']

my_mae = mine_sample['mae']
ptu_mae = ptu_sample['MAE']

# Perform paired t-test for MAE
mae_t_stat, mae_p_value = stats.ttest_rel(my_mae, ptu_mae)

print("T-stat MAE: " + str(mae_t_stat))
print("P-value MAE: " + str(mae_p_value))

# Perform paired t-test for RMSE
rmse_t_stat, rmse_p_value = stats.ttest_rel(my_rmse, ptu_rmse)

print("T-stat RMSE: " + str(rmse_t_stat))
print("P-value RMSE: " + str(rmse_p_value))






