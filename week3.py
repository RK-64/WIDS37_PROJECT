import pandas as pd
import numpy as np
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer

# Load master dataset
master_dataset = pd.read_csv("master_dataset.csv")
print("Dataset loaded:", master_dataset.shape)

# Convert stringified lists to Python objects
features = ['cast', 'crew', 'keywords', 'genres']

for feature in features:
    master_dataset[feature] = master_dataset[feature].apply(
        lambda x: literal_eval(x) if isinstance(x, str) else []
    )

print("Converted string columns to Python objects")
# Extract director from crew
def get_director(crew_list):
    for person in crew_list:
        if person.get('job') == 'Director':
            return person.get('name')
    return np.nan

master_dataset['director'] = master_dataset['crew'].apply(get_director)


# Extract top 3 cast members
master_dataset['cast'] = master_dataset['cast'].apply(
    lambda x: [i.get('name') for i in x][:3] if isinstance(x, list) else []
)


# Extract keyword names
master_dataset['keywords'] = master_dataset['keywords'].apply(
    lambda x: [i.get('name') for i in x] if isinstance(x, list) else []
)


# Extract genre names
master_dataset['genres'] = master_dataset['genres'].apply(
    lambda x: [i.get('name') for i in x] if isinstance(x, list) else []
)


# Normalize text (lowercase + remove spaces)
def clean_text(text_list):
    return [str(i).lower().replace(" ", "") for i in text_list]

master_dataset['cast'] = master_dataset['cast'].apply(clean_text)
master_dataset['keywords'] = master_dataset['keywords'].apply(clean_text)
master_dataset['genres'] = master_dataset['genres'].apply(clean_text)

master_dataset['director'] = master_dataset['director'].astype(str)
master_dataset['director'] = master_dataset['director'].apply(
    lambda x: [x.lower().replace(" ", "")] * 3
)


# Stem keywords
stemmer = SnowballStemmer('english')

master_dataset['keywords'] = master_dataset['keywords'].apply(
    lambda x: [stemmer.stem(i) for i in x]
)

# Create the "soup" feature
master_dataset['soup'] = (
    master_dataset['keywords'] +
    master_dataset['cast'] +
    master_dataset['director'] +
    master_dataset['genres']
)

master_dataset['soup'] = master_dataset['soup'].apply(lambda x: ' '.join(x))

print("Soup feature created")


# Save Week 3 output
master_dataset.to_csv("master_dataset_new.csv", index=False)

print("Week 3 processing complete")
print("Final dataset shape:", master_dataset.shape)
