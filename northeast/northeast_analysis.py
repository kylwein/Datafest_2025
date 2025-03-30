import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)
pd.set_option('display.float_format', lambda x: f'{x:,.0f}')


leases_df = pd.read_csv("../Leases.csv")
occupancy_df = pd.read_csv("../Major Market Occupancy Data-revised.csv")
price_df = pd.read_csv("../Price and Availability Data.csv")
unemployment_df = pd.read_csv("../Unemployment.csv")


'''
Let's look at the northwest region, focusing on market summary, time, etc..
'''

northeast_df = leases_df[leases_df['region'] == 'Northeast']

market_summary = northeast_df.groupby('market').agg(
    transactions=('leasedSF', 'count'),
    avg_leasedSF=('leasedSF', 'mean'),
    total_leasing=('leasing', 'sum')
).reset_index()
print("Market Summary:")
print(market_summary.to_string(index=False))

time_summary = northeast_df.groupby(['year', 'quarter']).agg(
    total_leasing=('leasing', 'sum'),
    avg_leasedSF=('leasedSF', 'mean')
).reset_index()
print("\nTime Trends (Leasing)")
print(time_summary.to_string(index=False))




#TIME PLOT
plt.figure(figsize=(10, 6))
sns.lineplot(data=time_summary, x='year', y='total_leasing', hue='quarter', marker='o')
plt.title("Total Leasing in Northeast Over Time")
plt.xlabel("Year")
plt.ylabel("Total Leasing (sq ft)")
plt.tight_layout()
plt.show()


# SPACE VS LEASING PLOT
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=northeast_df,
    x='available_space',
    y='leasing',
    hue='market',
    palette="deep",
    alpha=0.7
)

plt.xlabel("Available Space (in millions of sq ft)")
plt.ylabel("Leasing (in millions of sq ft)")
plt.title("Available Space vs. Leasing by Market (Northeast)")
plt.xticks(ticks=plt.xticks()[0], labels=[f"{x/1e6:.1f}" for x in plt.xticks()[0]])
plt.yticks(ticks=plt.yticks()[0], labels=[f"{y/1e6:.1f}" for y in plt.yticks()[0]])
plt.tight_layout()
plt.show()



print('--------------------------------------------------------------------------------------\n')
'''
Comparing the mean leased square footage (leasedSF) between Class A and Class O properties in the Northeast region. 

Class A: Premium buildings

Class O: Lower quality buildings
'''
# Separate the leasedSF for Class A and Class O
classA_leased = northeast_df[northeast_df['internal_class'] == 'A']['leasedSF'].dropna()
classO_leased = northeast_df[northeast_df['internal_class'] == 'O']['leasedSF'].dropna()

# Perform an independent two-sample t-test (assuming unequal variances)
t_stat, p_val = stats.ttest_ind(classA_leased, classO_leased, equal_var=False)

print("Two-sample t-test for leasedSF between Class A and Class O:")
print("T-val:", t_stat)
print("P-value:", p_val)


print('--------------------------------------------------------------------------------------\n')


