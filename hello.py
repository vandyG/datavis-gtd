import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('data/region_08.csv')

# Basic data exploration
def explore_data(df):
    print("Dataset Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\nBasic Statistics:\n", df.describe())

# Analyze temporal patterns
def temporal_analysis(df):
    monthly_attacks = df['imonth'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    monthly_attacks.plot(kind='bar')
    plt.title('Number of Attacks by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Attacks')
    plt.show()

# Analyze attack types and targets
def attack_target_analysis(df):
    # Attack types
    attack_types = df['attacktype1_txt'].value_counts()
    plt.figure(figsize=(12, 6))
    attack_types.plot(kind='bar')
    plt.title('Distribution of Attack Types')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Target types
    target_types = df['targtype1_txt'].value_counts()
    plt.figure(figsize=(12, 6))
    target_types.plot(kind='bar')
    plt.title('Distribution of Target Types')
    plt.xlabel('Target Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Geographical analysis
def geographical_analysis(df):
    country_attacks = df['country_txt'].value_counts()
    plt.figure(figsize=(12, 6))
    country_attacks.plot(kind='bar')
    plt.title('Number of Attacks by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Attacks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Analyze casualties
def casualty_analysis(df):
    # Create summary statistics for casualties
    casualties_summary = {
        'Total Killed': df['nkill'].sum(),
        'Average Killed per Attack': df['nkill'].mean(),
        'Max Killed in Single Attack': df['nkill'].max(),
        'Total Wounded': df['nwound'].sum(),
        'Average Wounded per Attack': df['nwound'].mean(),
        'Max Wounded in Single Attack': df['nwound'].max()
    }
    return pd.Series(casualties_summary)

# Analyze weapon types
def weapon_analysis(df):
    weapon_types = df['weaptype1_txt'].value_counts()
    plt.figure(figsize=(12, 6))
    weapon_types.plot(kind='bar')
    plt.title('Distribution of Weapon Types')
    plt.xlabel('Weapon Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Execute analysis
print("=== Basic Data Exploration ===")
explore_data(df)

print("\n=== Temporal Analysis ===")
temporal_analysis(df)

print("\n=== Attack and Target Analysis ===")
attack_target_analysis(df)

print("\n=== Geographical Analysis ===")
geographical_analysis(df)

print("\n=== Casualty Analysis ===")
print(casualty_analysis(df))

print("\n=== Weapon Analysis ===")
weapon_analysis(df)

# Additional insights
print("\n=== Success Rate Analysis ===")
success_rate = (df['success'].mean() * 100)
print(f"Attack Success Rate: {success_rate:.2f}%")

print("\n=== Property Damage Analysis ===")
property_damage = df['property'].value_counts()
print("Property Damage Distribution:\n", property_damage)

# Correlation analysis for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()