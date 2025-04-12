import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set Seaborn style
sns.set(style="darkgrid")

df = pd.read_csv("excel.csv")  
df.info()
df.head()

# Convert last_update to datetime
df['last_update'] = pd.to_datetime(df['last_update'])
df['date'] = df['last_update'].dt.date

# Drop rows with missing pollution data
df_clean = df.dropna(subset=['pollutant_avg'])

# Basic null check
df.isnull().sum()

df_clean.describe()

top_cities_max = df_clean.groupby('city')['pollutant_max'].max().nlargest(10)

sns.barplot(x=top_cities_max.values, y=top_cities_max.index, palette='RdYlGn')
plt.title("Top 10 Most Polluted Cities (by Max Pollution Level)")
plt.xlabel("Maximum Pollutant Level")
plt.show()


df = pd.read_csv("excel.csv")  

df_clean = df[df['pollutant_avg'].notna()]

def categorize_pollution(avg):
    if avg <= 50:
        return 'Low'
    elif avg <= 100:
        return 'Moderate'
    else:
        return 'High'

df_clean['pollution_level'] = df_clean['pollutant_avg'].apply(categorize_pollution)

custom_palette = {
    'Low': 'green',
    'Moderate': 'yellow',
    'High': 'red'
}

target_pollutants = ['NO2', 'OZONE', 'NH3', 'CO']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, pollutant in enumerate(target_pollutants):
    data = df_clean[df_clean['pollutant_id'].str.upper() == pollutant]
    
    sns.histplot(
        data=data,
        x='pollutant_avg',
        hue='pollution_level',
        bins=30,
        kde=True,
        palette=custom_palette,
        ax=axes[i]
    )
    axes[i].set_title(f"{pollutant} - Pollutant Average Distribution")
    axes[i].set_xlabel("Pollutant Average")
    axes[i].set_ylabel("Frequency")
    axes[i].legend(title='Pollution Level')

plt.tight_layout()
plt.show()


numeric_data = df_clean.select_dtypes(include='number')

corr = numeric_data.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=0.5)


plt.title(' Correlation Heatmap of Pollution Data', fontsize=14)
plt.show()

plt.figure(figsize=(12, 6))

# Boxplot of pollutant_avg by pollutant_id
sns.boxplot(x='pollutant_id', y='pollutant_avg', data=df_clean, palette='Pastel1', showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 6})

# Add colored stripplot for each state
sns.stripplot(x='pollutant_id', y='pollutant_avg', data=df_clean, hue='state',
              jitter=True, dodge=True, size=3, alpha=0.6)

plt.title("Pollution Distribution by Pollutant Across States")
plt.xticks(rotation=45)
plt.ylabel("Average Pollutant Level")
plt.xlabel("Pollutant Type")
plt.legend(title="State", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.show()

df = pd.read_csv("excel.csv") 


df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df.dropna(subset=['last_update'], inplace=True)


df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')
df.dropna(subset=['pollutant_avg'], inplace=True)

# Extract date
df['date'] = df['last_update'].dt.date

# Filter for Delhi only
delhi = df[df['city'] == 'Delhi']

# Plot daily pollutant trends in Delhi with markers
plt.figure(figsize=(12, 6))
for pollutant in delhi['pollutant_id'].unique():
    pollutant_df = delhi[delhi['pollutant_id'] == pollutant]
    daily_avg = pollutant_df.groupby('date')['pollutant_avg'].mean()
    plt.plot(daily_avg.index, daily_avg.values, marker='o', label=pollutant)

plt.title("Daily Pollutant Trends in Delhi with Markers")
plt.xlabel("Date")
plt.ylabel("Average Pollutant Level")
plt.legend(title="Pollutant")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("delhi_pollutants_with_markers.png")
plt.show()

# Filter data for Delhi and Mumbai
delhi_data = df[df['city'] == 'Delhi']['pollutant_avg']
mumbai_data = df[df['city'] == 'Mumbai']['pollutant_avg']

# Drop NaNs just in case
delhi_data = delhi_data.dropna()
mumbai_data = mumbai_data.dropna()

# Perform independent t-test
t_stat, p_val = stats.ttest_ind(delhi_data, mumbai_data, equal_var=False)

# Print test result
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_val:.4f}")
if p_val < 0.05:
    print("Result: Significant difference in pollution levels (p < 0.05)")
else:
    print("Result: No significant difference (p >= 0.05)")

# Boxplot to visualize the comparison
plt.figure(figsize=(8, 6))
sns.boxplot(data=[delhi_data, mumbai_data], palette=['red', 'blue'])
plt.xticks([0, 1], ['Delhi', 'Mumbai'])
plt.title("Comparison of Average Pollution Levels: Delhi vs Mumbai")
plt.ylabel("Pollutant Average")
plt.grid(True)
plt.tight_layout()
plt.show()
