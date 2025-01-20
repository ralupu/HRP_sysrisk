import pandas as pd
import matplotlib.pyplot as plt

# 1. Preprocessing
# Load the Excel file containing the CoVaR data
file_path = 'data/ResultResults_S600_oct24.xlsx'
sheet_name = 'CoVaR (K=95%)'

# Load the sheet into a pandas DataFrame
covar_data = pd.read_excel(file_path, sheet_name=sheet_name)

# Convert Date column to datetime and set it as index
covar_data['Date'] = pd.to_datetime(covar_data['Date'], format='%d/%m/%Y')
covar_data.set_index('Date', inplace=True)

# Load the Excel file with Instrument, Company Name, and GICS Industry Group Name
file_path = 'data/names_sectors.xlsx'
names_sectors = pd.read_excel(file_path)

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
plt.title('Average CoVaR Comparison: Household & Personal Products vs. Banks')
plt.xlabel('Date')
plt.ylabel('Average CoVaR')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
# Show the plot
plt.show()