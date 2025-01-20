import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

# 1. Preprocessing
# Load the Excel file containing the CoVaR data
file_path = 'data/ResultResults_S600_oct24.xlsx'
sheet_name = 'CoVaR (K=95%)'

# Load the sheet into a pandas DataFrame
covar_data = pd.read_excel(file_path, sheet_name=sheet_name)

# Convert Date column to datetime and set it as index
covar_data['Date'] = pd.to_datetime(covar_data['Date'], format='%d/%m/%Y')
covar_data.set_index('Date', inplace=True)

# 2. Correlation Matrix
# Compute the correlation matrix of CoVaR values for the assets
correlation_matrix = covar_data.corr()

# Clean the correlation matrix by replacing NaN and infinite values with 0
cleaned_correlation_matrix = correlation_matrix.replace([np.inf, -np.inf], 0).fillna(0)

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))  # Adjust figure size as needed
sns.heatmap(cleaned_correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
# plt.title('Heatmap of Correlation Matrix')
output_file_path = 'heatmap_corr_matrix.png'
plt.savefig(output_file_path, format='png', dpi=300)
plt.show()

# 3. Hierarchical Clustering
# Perform hierarchical clustering using the 'ward' method on the cleaned correlation matrix
linkage_matrix = linkage(1 - cleaned_correlation_matrix, method='ward')

# Plot a dendrogram to visualize the clustering without labels
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, leaf_rotation=90, no_labels=True)  # no_labels=True to remove labels
# plt.title('Dendrogram of Assets using HRP')
plt.xlabel('Assets')
plt.ylabel('Distance')
output_file_path = 'dedogram_using_HRP.png'
plt.savefig(output_file_path, format='png', dpi=300)
plt.show()

# 4. Quasi-Diagonalization
# Get the order of assets after clustering
ordered_assets = leaves_list(linkage_matrix)

# Reorder the correlation matrix based on the clustering result
reordered_correlation_matrix = cleaned_correlation_matrix.iloc[ordered_assets, ordered_assets]

# Plot the heatmap of the reordered correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(reordered_correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
# plt.title('Heatmap of Reordered Correlation Matrix (HRP)')
output_file_path = 'heatmap_using_HRP.png'
plt.savefig(output_file_path, format='png', dpi=300)
plt.show()


# 5. Recursive Bisection Allocation
def get_cluster_var(cov_matrix, assets):
    """
    Compute the variance of a cluster of assets based on the covariance matrix.
    """
    cov_sub = cov_matrix.loc[assets, assets]
    w = np.ones(len(assets)) / len(assets)  # Equal weight allocation for the assets
    return np.dot(w, np.dot(cov_sub, w))


def recursive_bisection(cov_matrix, assets):
    """
    Recursively allocate risk using the HRP methodology.
    """
    # If only one asset is left, allocate 100% of risk to it
    if len(assets) == 1:
        return {assets[0]: 1.0}

    # Split the assets into two clusters
    split = len(assets) // 2
    cluster_1 = assets[:split]
    cluster_2 = assets[split:]

    # Calculate the variance for each cluster
    var_1 = get_cluster_var(cov_matrix, cluster_1)
    var_2 = get_cluster_var(cov_matrix, cluster_2)

    # Allocate risk proportional to the inverse variance of each cluster
    total_var = var_1 + var_2
    alloc_1 = 1 - var_1 / total_var
    alloc_2 = 1 - var_2 / total_var

    # Recursively allocate within each cluster
    allocation = {}
    allocation.update({k: v * alloc_1 for k, v in recursive_bisection(cov_matrix, cluster_1).items()})
    allocation.update({k: v * alloc_2 for k, v in recursive_bisection(cov_matrix, cluster_2).items()})

    return allocation


# Apply the recursive bisection method to allocate risk based on the reordered correlation matrix
assets_ordered = reordered_correlation_matrix.columns
allocation = recursive_bisection(reordered_correlation_matrix, assets_ordered)

# Display the sorted allocation result
allocation_sorted = dict(sorted(allocation.items(), key=lambda item: item[1], reverse=True))
print(allocation_sorted)

# 6. HRP-Based CoVaR Calculation
# You can use the allocated weights (in 'allocation_sorted') to calculate CoVaR based on HRP
# For example, you can compute a weighted average of CoVaR values for each asset, using the allocation weights
hrp_covar = covar_data[assets_ordered].mul([allocation_sorted[asset] for asset in assets_ordered], axis=1).sum(axis=1)

# Display the HRP-based CoVaR results
print(hrp_covar.head())


# Convert the allocation dictionary to two lists: one for assets, one for allocations
assets = list(allocation_sorted.keys())
allocations = list(allocation_sorted.values())

# Create a horizontal bar chart
plt.figure(figsize=(12, 8))  # Adjust size as needed
plt.barh(assets, allocations, color='skyblue')
plt.xlabel('Risk Allocation')
plt.ylabel('Assets')
# plt.title('HRP-Based Risk Allocation for Assets')
# Show only every 10th asset label
plt.yticks(ticks=np.arange(0, len(assets), 10), labels=[assets[i] for i in range(0, len(assets), 10)])
# Invert y-axis to have the highest allocation at the top
plt.gca().invert_yaxis()
# Adjust layout for better spacing
plt.tight_layout()

output_file_path = 'HRP_risk_allocation.png'
plt.savefig(output_file_path, format='png', dpi=300)
# Display the chart
plt.show()



# Assuming 'covar_data' contains the initial CoVaR values and 'hrp_covar' contains the final HRP-based CoVaR

# Compute the difference between initial and HRP-based CoVaR
covar_difference = covar_data.mean(axis=1) - hrp_covar

# Plot a bar chart of the differences
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.bar(covar_data.index, covar_difference, color='lightcoral')
plt.xlabel('Date')
plt.ylabel('Difference in CoVaR')
# plt.title('Difference between Initial and HRP-Based CoVaR')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
output_file_path = 'difference_in_covar.png'
plt.savefig(output_file_path, format='png', dpi=300)
# Show the chart
plt.show()


# Assuming 'covar_data' contains the initial CoVaR values and 'hrp_covar' contains the final HRP-based CoVaR

# Compute the mean CoVaR for each date (initial CoVaR values)
mean_initial_covar = covar_data.mean(axis=1)

# Plot initial and HRP-based CoVaR values on the same plot
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.plot(covar_data.index, mean_initial_covar, label='Initial CoVaR', color='blue', linestyle='--')
plt.plot(covar_data.index, hrp_covar, label='HRP-Based CoVaR', color='green')
plt.xlabel('Date')
plt.ylabel('CoVaR Value')
# plt.title('Comparison of Initial and HRP-Based CoVaR')
plt.legend(loc='upper left')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
output_file_path = 'comparison_covars.png'
plt.savefig(output_file_path, format='png', dpi=300)
# Show the chart
plt.show()



# 1. Ensure that the allocation_sorted and covar_data are aligned with the company names
# Strip any extra spaces from company names
covar_data.columns = covar_data.columns.str.strip()

# Ensure the company names in allocation_sorted match the columns in covar_data
allocation_sorted = {k.strip(): v for k, v in allocation_sorted.items()}

# 2. Initialize a dictionary to store the weighted CoVaR for each company
weighted_hrp_covar = {}

# 3. Iterate over each company in covar_data and apply the weights from allocation_sorted
for company in covar_data.columns:
    if company in allocation_sorted:  # Only proceed if the company exists in both
        # Compute the weighted CoVaR for each company using the HRP weight
        weighted_hrp_covar[company] = covar_data[company] * allocation_sorted[company]

# 4. Aggregate the weighted CoVaRs for each day to compute a single CoVaR for each company
# Sum across the weighted CoVaRs to get the final HRP-based CoVaR for each day
final_hrp_covar = pd.DataFrame(weighted_hrp_covar).sum(axis=1)



# Testing
# 1. Preprocessing to get daily returns for the STOXX 600 and all companies
initial_prices = pd.read_excel('data/InitialPricesAllCompanies.xlsx')
returns = initial_prices.iloc[:, 1:].pct_change().dropna()

# 1. Preprocessing: ensure that returns and covar_data have aligned dates and columns
returns.columns = returns.columns.str.strip()
covar_data.columns = covar_data.columns.str.strip()

# Ensure that the dates match between returns and covar_data
covar_data = covar_data.reindex(returns.index)

# 2. Initialize a dictionary to store violation rates for each company
violation_rates = {}

# 3. Iterate over each company (column) in covar_data and compute violation rates
for company in covar_data.columns:
    if company in returns.columns:  # Only proceed if the company exists in both DataFrames
        # For each company, check if the company is in distress (return < VaR)
        company_distress = returns[company] < returns[company].quantile(0.05)  # Company's own VaR (5th percentile)

        # Check if, on the same day, the STOXX return is below the CoVaR for that company
        stoxx_violation = returns['STOXX'] < covar_data[company]

        # Count where both conditions are true (company in distress AND STOXX return < CoVaR)
        violations = (company_distress & stoxx_violation).sum()

        # Count the total number of distress events for this company
        total_conditions = company_distress.sum()

        # Calculate the violation rate and store it in the dictionary
        violation_rate = violations / total_conditions if total_conditions > 0 else 0
        violation_rates[company] = violation_rate

# 4. Convert the violation rates dictionary to a pandas DataFrame for easier plotting
violation_rates_df = pd.DataFrame(list(violation_rates.items()), columns=['Company', 'Violation Rate'])

# Sort the DataFrame by violation rate for better visualization
violation_rates_df = violation_rates_df.sort_values(by='Violation Rate', ascending=False)
violation_rates_df.to_excel('results/violation_rates.xlsx')

# # 5. Plot the violation rates as a bar chart
# plt.figure(figsize=(12, 8))
# plt.barh(violation_rates_df['Company'], violation_rates_df['Violation Rate'], color='skyblue')
# plt.xlabel('Violation Rate (%)')
# plt.title('CoVaR Violation Rates for All Companies')
# plt.gca().invert_yaxis()  # Invert y-axis to show highest violation rate at the top
# plt.tight_layout()
#
# # Show the plot
# plt.show()


# # 1. Preprocessing: Ensure that returns and hrp_covar have aligned dates and columns
# returns.columns = returns.columns.str.strip()
# final_hrp_covar.columns = final_hrp_covar.columns.str.strip()
#
# # Ensure that the dates match between returns and hrp_covar
# final_hrp_covar = final_hrp_covar.reindex(returns.index)
#
# # 2. Initialize a dictionary to store HRP-based violation rates for each company
# hrp_violation_rates = {}
#
# # 3. Iterate over each company (column) in final_hrp_covar and compute violation rates
# for company in final_hrp_covar.columns:
#     if company in returns.columns:  # Only proceed if the company exists in both DataFrames
#         # For each company, check if the company is in distress (return < VaR)
#         company_distress = returns[company] < returns[company].quantile(0.05)  # Company's own VaR (5th percentile)
#
#         # Check if, on the same day, the STOXX return is below the HRP-based CoVaR for that company
#         stoxx_violation_hrp = returns['STOXX'] < final_hrp_covar[company]
#
#         # Count where both conditions are true (company in distress AND STOXX return < HRP-based CoVaR)
#         hrp_violations = (company_distress & stoxx_violation_hrp).sum()
#
#         # Count the total number of distress events for this company
#         total_conditions = company_distress.sum()
#
#         # Calculate the violation rate and store it in the dictionary
#         hrp_violation_rate = hrp_violations / total_conditions if total_conditions > 0 else 0
#         hrp_violation_rates[company] = hrp_violation_rate
#
# # 4. Convert the HRP-based violation rates dictionary to a pandas DataFrame for easier plotting
# hrp_violation_rates_df = pd.DataFrame(list(hrp_violation_rates.items()), columns=['Company', 'HRP Violation Rate'])
#
# # Sort the DataFrame by HRP violation rate for better visualization
# hrp_violation_rates_df = hrp_violation_rates_df.sort_values(by='HRP Violation Rate', ascending=False)
# hrp_violation_rates_df.to_excel('results/hrp_violation_rates.xlsx')
#
# # 5. Plot the HRP-based violation rates as a bar chart
# plt.figure(figsize=(12, 8))
# plt.barh(hrp_violation_rates_df['Company'], hrp_violation_rates_df['HRP Violation Rate'], color='lightcoral')
# plt.xlabel('HRP Violation Rate (%)')
# plt.title('HRP-Based CoVaR Violation Rates for All Companies')
# plt.gca().invert_yaxis()  # Invert y-axis to show highest violation rate at the top
# plt.tight_layout()
#
# # Show the plot
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

# 1. Sort the HRP allocations (stored in 'allocation_sorted') from highest to lowest
sorted_allocations = sorted(allocation_sorted.values(), reverse=True)
sorted_companies = [k for k, v in sorted(allocation_sorted.items(), key=lambda item: item[1], reverse=True)]

# 2. Plot the sorted allocations to visualize the elbow
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(sorted_allocations) + 1), sorted_allocations, marker='o')
plt.xlabel('Company Rank')
plt.ylabel('HRP Allocation')
# plt.title('Elbow Rule for HRP-Based Allocations')
plt.grid(True)
plt.tight_layout()
output_file_path = 'elbow_rule.png'
plt.savefig(output_file_path, format='png', dpi=300)
plt.show()

# 3. Use the KneeLocator from the kneed package to find the elbow point
knee_locator = KneeLocator(range(1, len(sorted_allocations) + 1), sorted_allocations, curve='convex', direction='decreasing')

# Get the index of the elbow point
elbow_point = knee_locator.elbow

# Print the elbow point and the corresponding companies
print(f"Elbow point found at company rank: {elbow_point}")
top_companies = sorted_companies[:elbow_point]
print(f"Companies with the highest allocations: {top_companies}")

# Optional: Highlight the elbow point on the plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(sorted_allocations) + 1), sorted_allocations, marker='o')
plt.axvline(x=elbow_point, color='red', linestyle='--', label=f'Elbow at {elbow_point}')
plt.xlabel('Company Rank')
plt.ylabel('HRP Allocation')
# plt.title('Elbow Rule for HRP-Based Allocations')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_file_path = 'elbow_rule_HRP.png'
plt.savefig(output_file_path, format='png', dpi=300)
plt.show()

names_sectors = pd.read_excel('data/names_sectors.xlsx')

# names_sectors.Instrument = names_sectors.Instrument.str.replace('.', '_')
# names_sectors.Instrument = [col.lstrip('_') if col.startswith('_') else col for col in names_sectors.Instrument]
# names_sectors.Instrument = names_sectors.Instrument.str.split('_').str[0]

sectors = [names_sectors[names_sectors['Instrument'] == company]['GICS Industry Group Name'].values[0] for company in top_companies]
names = [names_sectors[names_sectors['Instrument'] == company]['Company Name'].values[0] for company in top_companies]
# values =
print(names)
print(sectors)

dict = {names[i]: sectors[i] for i in range(len(sectors))}
top_companies_df = pd.DataFrame(list(dict.items()), columns=['Company Name', 'Sector'])
values = [float(allocation_sorted[company]) for company in top_companies]
top_companies_df['Allocations'] = values
top_companies_df.to_excel('results/top_companies.xlsx')


names_sectors['Instrument'] = names_sectors['Instrument'].str.strip()

# Create a DataFrame from the 'allocation_sorted' dictionary
allocations_df = pd.DataFrame(list(allocation_sorted.items()), columns=['Instrument', 'Allocation'])

# Merge the allocation data with the sector information based on the 'Instrument' column
merged_data = pd.merge(names_sectors, allocations_df, on='Instrument', how='inner')

# Compute the average allocation for each sector (GICS Industry Group Name)
average_allocation_by_sector = merged_data.groupby('GICS Industry Group Name')['Allocation'].mean()

average_allocation_by_sector_sorted = average_allocation_by_sector.sort_values(ascending=False)
average_allocation_by_sector_sorted.to_excel('results/average_allocation_by_sector.xlsx')



covar_data.to_pickle('results/covar_data.pickle')
# Ensure the 'Instrument' column is aligned with the 'covar_data' column names (strip whitespace if needed)
names_sectors['Instrument'] = names_sectors['Instrument'].str.strip()
covar_data.columns = covar_data.columns.str.strip()

# Extract Household & Personal Products and Banks companies
household_companies = set(names_sectors[names_sectors['GICS Industry Group Name'] == 'Household & Personal Products']['Instrument'])
banks_companies = set(names_sectors[names_sectors['GICS Industry Group Name'] == 'Banks']['Instrument'])

print(household_companies)
# Filter the CoVaR data for Household & Personal Products and Banks companies
household_covar = covar_data.loc[:,list(household_companies)]
banks_covar = covar_data.loc[:,list(banks_companies)]

# Compute the average CoVaR for Household & Personal Products sector
avg_covar_household = household_covar.mean(axis=1)

# Compute the average CoVaR for Banks sector
avg_covar_banks = banks_covar.mean(axis=1)

# Plot the comparison
plt.figure(figsize=(10, 6))
plt.plot(covar_data.index, avg_covar_household, label='Household & Personal Products', color='blue')
plt.plot(covar_data.index, avg_covar_banks, label='Banks', color='red')
# plt.title('Average CoVaR Comparison: Household & Personal Products vs. Banks')
plt.xlabel('Date')
plt.ylabel('Average CoVaR')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

output_file_path = 'household_banks_comparison.png'
plt.savefig(output_file_path, format='png', dpi=300)
# Show the plot
plt.show()


