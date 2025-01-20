import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'data/ResultResults_S600_oct24.xlsx'
sheet_name = 'CoVaR (K=95%)'

# Load the sheet into a pandas DataFrame
covar_data = pd.read_excel(file_path, sheet_name=sheet_name)

# Melting the CoVaR data for easier manipulation
covar_data_melted = covar_data.melt(id_vars='Date', var_name='Company', value_name='CoVaR')

data = pd.read_excel('data/names_sectors.xlsx')

# Merging the melted CoVaR data with the sector mapping
sector_mapping = data[['Instrument', 'GICS Industry Group Name']].rename(
    columns={'Instrument': 'Company', 'GICS Industry Group Name': 'Sector'}
)
covar_with_sector = pd.merge(covar_data_melted, sector_mapping, on='Company', how='left')

# Drop any rows with missing sector information
covar_with_sector.dropna(subset=['Sector'], inplace=True)



# Plotting
plt.figure(figsize=(14, 7))
sns.boxplot(data=covar_with_sector, x='Sector', y='CoVaR')
plt.xticks(rotation=90)
# plt.title('Boxplot of CoVaR Values Grouped by Sector')
plt.xlabel('Sector')
plt.ylabel('CoVaR Value')
plt.tight_layout()
output_file_path = 'covar_boxplot_by_sector.png'
plt.savefig(output_file_path, format='png', dpi=300)
plt.show()
