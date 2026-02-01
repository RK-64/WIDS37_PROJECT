import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("master_dataset_final.csv")
print("Dataset loaded:", df.shape)


# 2️ Extract Director Name Correctly 
import ast

def decode_director(x):
    try:
        director_list = ast.literal_eval(x)
        if isinstance(director_list, list) and len(director_list) > 0:
            return director_list[0].replace(" ", "").title()
    except:
        pass
    return "Unknown"

if 'director' in df.columns:
    df['Director'] = df['director'].apply(decode_director)
else:
    df['Director'] = "Unknown"



# 3️ Clean Movie Titles 

df['title_clean'] = df['title'].str.lower().str.strip()

# 4️ Count Vectorizer 
count = CountVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english'
)

count_matrix = count.fit_transform(df['soup'])
print("Count matrix shape:", count_matrix.shape)


# 5️ Cosine Similarity Computation

cosine_sim = cosine_similarity(count_matrix, count_matrix)
print("Cosine similarity matrix created")

# 6️ Index Mapping
indices = pd.Series(df.index, index=df['title_clean']).drop_duplicates()
# 7️ Recommendation Function
def recommend_movies(title, top_n=10):
    title = title.lower().strip()

    if title not in indices:
        print(" Movie not found in dataset")
        return None

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    result = df.iloc[movie_indices][
        ['title', 'Director', 'release_date']
    ]

    result.columns = ['Movie Title', 'Director', 'Release Date']
    return result

# 8️ Example Output: Avatar
output = recommend_movies("Avatar", top_n=12)

if output is not None:
    print("\n Recommended Movies for Avatar\n")
    print(output.to_string(index=False))
print("finished wids hehe !!!!")