import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6] # Default figure size

print("--- TASK 1: Initial Data Exploration & Inspection ---")

try:
    df = pd.read_csv('netflix_titles.csv')
    total_initial_rows = df.shape[0]

    print("\n1. Preview of the first 10 rows:")
    print(df.head(10))

    df.drop_duplicates(subset=['show_id'], inplace=True)
    rows_dropped = total_initial_rows - df.shape[0]

    df.drop('description', axis=1, inplace=True)

    print(f"\n2. Rows Processed: {total_initial_rows}, Duplicates Dropped: {rows_dropped}")
    print(f"3. Final Dataset Shape after cleaning: {df.shape}")

    print("\n4. Final Data Types & Non-Null Counts :")
    df.info()

    print("\n5. Basic Statistics :")
    print(df.describe(include='all').to_string())

    print("\n6. Missing Value Summary:")
    missing_info = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing Percent': (df.isnull().sum() / len(df)) * 100
    })
    missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(
        by='Missing Count', ascending=False
    )
    print(missing_info.to_string(float_format="{:.2f}".format))
    print("\n*** Task 1 Complete. ***")


    print("\n--- TASK 2: Data Cleaning & Feature Engineering ---")

    df['country'].fillna('Unknown', inplace=True)

    df['director'].fillna('No Director Listed', inplace=True)

    df['cast'].fillna('No Cast Listed', inplace=True)
    df['rating'].fillna(df['rating'].mode()[0], inplace=True)
    df['duration'].fillna(df['duration'].mode()[0], inplace=True)
    df['date_added'].fillna('Missing Date', inplace=True)

    print("1. Missing Values After Imputation (Should all be 0 or near 0):")
    print(df.isnull().sum().to_string())



    df['duration_minutes'] = np.nan
    df['seasons'] = np.nan

    movie_mask = df['type'] == 'Movie'
    df.loc[movie_mask, 'duration_minutes'] = df.loc[movie_mask, 'duration'].str.extract(r'(\d+) min').astype(float)

    tv_mask = df['type'] == 'TV Show'
    df.loc[tv_mask, 'seasons'] = df.loc[tv_mask, 'duration'].str.extract(r'(\d+) Season').astype(float)

    print("\n2. Preview of New Duration Features:")
    print(df[['type', 'duration', 'duration_minutes', 'seasons']].sample(5).to_string())


    df['Is_Recent'] = np.where(df['release_year'] >= 2015, 1, 0)

    print("\n3. Preview of New Binary Feature (Is_Recent):")
    print(df[['release_year', 'Is_Recent']].sample(5).to_string())

    print("\n*** Task 2 Complete. ***")
    
    
    print("\n--- TASK 3: ---")
    
    plt.figure(figsize=(7, 5))
    sns.countplot(x='type', data=df)
    plt.title('Content Type Distribution (Movies vs TV Shows)')
    plt.xlabel('Content Type')
    plt.ylabel('Count')
    print("\n1. Count Plot: Content Type Distribution ")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['release_year'], bins=30, kde=True)
    plt.title('Distribution of Content Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Count')
    print("2. Histogram: Distribution of Content Release Year ")
    plt.show()
    
    country_df = df['country'].str.split(',', expand=True).stack().str.strip()
    top_10_countries = country_df.value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_countries.index, y=top_10_countries.values, palette='viridis')
    plt.title('Top 10 Countries by Content Count')
    plt.xlabel('Country')
    plt.ylabel('Number of Releases')
    plt.xticks(rotation=45, ha='right')
    print("3. Bar Plot: Top 10 Countries by Content Count ")
    plt.tight_layout()
    plt.show()

    movie_duration_df = df[df['type'] == 'Movie'].dropna(subset=['duration_minutes'])

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Is_Recent', y='duration_minutes', data=movie_duration_df)
    plt.title('Movie Duration Comparison: Recent (>=2015) vs Older (<2015)')
    plt.xlabel('Is Recent (0: Older, 1: Recent)')
    plt.ylabel('Duration in Minutes')
    print("4. Box Plot: Movie Duration by Recent/Older Classification ")
    plt.show()


    numerical_features = ['release_year', 'duration_minutes', 'seasons', 'Is_Recent']
    corr_matrix = df[numerical_features].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Features')
    print("5. Correlation Heatmap ")
    plt.show()

  


    print("\n*** All Tasks Complete. ***")


except FileNotFoundError:
    print("Error: netflix_titles.csv not found. Please ensure the file is in the correct directory.")