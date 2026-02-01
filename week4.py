import pandas as pd
import numpy as np


#  Load Week-3 Output

df = pd.read_csv("master_dataset_new.csv")
print("Loaded dataset shape:", df.shape)


#  Drop Unused / Heavy Metadata Columns


columns_to_drop = [
    'adult', 'belongs_to_collection', 'budget', 'homepage',
    'original_language', 'production_companies',
    'production_countries', 'revenue', 'runtime',
    'spoken_languages', 'status', 'video',
    'overview', 'tagline', 'vote_average', 'vote_count',
    'cast', 'crew', 'keywords',
    'imdb_id', 'original_title', 'poster_path', 'genres'
]

df.drop(
    columns=[col for col in columns_to_drop if col in df.columns],
    inplace=True
)

print("After dropping columns:", df.shape)


#  Clean Popularity Column

df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df.dropna(subset=['popularity'], inplace=True)


# Sort Movies by Popularity
df.sort_values(by='popularity', ascending=False, inplace=True)

#  Reset Index
df.reset_index(drop=True, inplace=True)

#  Final Required Columns Check
df.dropna(subset=['title', 'soup'], inplace=True)
print("Final dataset shape:", df.shape)


#  Save Week-4 Output
df.to_csv("master_dataset_final.csv", index=False)

print(" Week-4 processing complete")
print("Saved as: master_dataset_final.csv")
