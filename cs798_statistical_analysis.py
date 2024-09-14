import pandas as pd
from scipy import stats
import random

def perform_ttest(domain, sample_100=False):
    """
    Function to perform t-tests for a given domain with an option to sample 100 users or use all overlapping users.
    
    :param domain: str, the domain to be used (e.g., "bookmovie", "electronicfood")
    :param sample_100: bool, whether to sample 100 users (True) or use all overlapping users (False)
    """
    # Load the PTUPCDR results
    ptu = pd.read_csv(f"./results_cs798/ptupcdr_mae_rmse_{domain}.csv")

    # List of other algorithm file names and their respective names
    other_algos = {
        "CMF": f"./results_cs798/cmf_mae_rmse_{domain}.csv",
        "TGT": f"./results_cs798/tgt_mae_rmse_{domain}.csv",
        "EMCDR": f"./results_cs798/emcdr_mae_rmse_{domain}.csv"
    }

    # List to store the results
    results_list = []

    # Merge PTUPCDR with the first algorithm to get a sample of users
    first_algo_file = list(other_algos.values())[0]
    first_algo_data = pd.read_csv(first_algo_file)
    merged = pd.merge(ptu, first_algo_data, on='UID', suffixes=('_ptu', '_first_algo'))

    # Optionally sample 100 users
    if sample_100:
        random_indices = random.sample(range(len(merged)), 100)
        merged_sample = merged.iloc[random_indices]
        sampled_uids = merged_sample['UID'].tolist()  # Keep track of sampled UIDs
    else:
        sampled_uids = merged['UID'].tolist()

    # Now loop through each algorithm, ensuring we use the same sampled UIDs
    for algo_name, algo_file in other_algos.items():
        # Load the other algorithm's results
        algo_data = pd.read_csv(algo_file)

        # Merge on 'UID' to align the data by users who exist in both datasets
        merged = pd.merge(ptu, algo_data, on='UID', suffixes=('_ptu', f'_{algo_name.lower()}'))

        # Filter merged data to only include the sampled UIDs
        merged_sample = merged[merged['UID'].isin(sampled_uids)]

        # Perform paired t-tests directly on the 'MAE' and 'RMSE' columns for overlapping users
        mae_t_stat, mae_p_value = stats.ttest_rel(merged_sample['MAE_ptu'], merged_sample[f'MAE_{algo_name.lower()}'])
        rmse_t_stat, rmse_p_value = stats.ttest_rel(merged_sample['RMSE_ptu'], merged_sample[f'RMSE_{algo_name.lower()}'])

        # Append the results including the number of overlapping users
        results_list.append({
            'Algorithm': algo_name,
            'Overlapping_Users': len(merged_sample),
            'MAE_t_stat': mae_t_stat,
            'MAE_p_value': mae_p_value,
            'RMSE_t_stat': rmse_t_stat,
            'RMSE_p_value': rmse_p_value
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results to a CSV file
    results_df.to_csv(f"./results_cs798/ttest_results_{domain}_{'sampled' if sample_100 else 'all_users'}.csv", index=False)

    print(f"Results saved to ttest_results_{domain}_{'sampled' if sample_100 else 'all_users'}.csv")


# Example usage:
# For Book-Movie with all users:
#perform_ttest(domain="bookmovie", sample_100=False)

# For Electronic-Food with sampling 100 users:
perform_ttest(domain="electronicfood", sample_100=False)

# Run the t-tests for Book-Food without sampling (all users)
perform_ttest(domain="bookfood", sample_100=False)

# Run the t-tests for Movie-Music with sampling 100 users
perform_ttest(domain="moviemusic", sample_100=False)
