import gradio as gr
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ================= LOAD PKL FILES =================

with open("movie_info.pkl", "rb") as f:
    movie_info_df = pickle.load(f)

with open("user_item_matrix.pkl", "rb") as f:
    user_item_matrix = pickle.load(f)

with open("item_similarity.pkl", "rb") as f:
    item_similarity_df = pickle.load(f)

# ================= USER-BASED =================

def user_based_recommend(user_id, top_n=10):
    user_id = int(user_id)

    if user_id not in user_item_matrix.index:
        return "User not found."

    user_similarity = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    sim_scores = user_sim_df.loc[user_id].drop(user_id)
    neighbors = sim_scores.sort_values(ascending=False).head(5).index

    neighbor_ratings = user_item_matrix.loc[neighbors]
    weights = sim_scores.loc[neighbors].values.reshape(-1, 1)

    scores = (neighbor_ratings.T @ weights).flatten()
    scores = scores / np.sum(np.abs(weights))

    preds = pd.Series(scores, index=user_item_matrix.columns)
    seen = user_item_matrix.loc[user_id]
    preds = preds[seen == 0]

    top_movies = preds.sort_values(ascending=False).head(top_n)

    result = movie_info_df[movie_info_df.movie_id.isin(top_movies.index)]
    result = result.set_index("movie_id").loc[top_movies.index]
    result["score"] = top_movies.values

    return result[["movie title", "score"]]

# ================= ITEM-BASED (USER ID) =================

def item_based_user(user_id, top_n=10):
    user_id = int(user_id)

    if user_id not in user_item_matrix.index:
        return "User not found."

    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]

    scores = pd.Series(0, index=user_item_matrix.columns)

    for movie_id, rating in rated_movies.items():
        scores += item_similarity_df[movie_id] * rating

    scores = scores.drop(rated_movies.index)
    top_movies = scores.sort_values(ascending=False).head(top_n)

    result = movie_info_df[movie_info_df.movie_id.isin(top_movies.index)]
    result = result.set_index("movie_id").loc[top_movies.index]
    result["score"] = top_movies.values

    return result[["movie title", "score"]]

# ================= MOVIE NAME BASED =================

def movie_name_based(movie_name, top_n=10):
    titles_clean = movie_info_df["movie title"].str.replace(
        r"\s*\(\d{4}\)", "", regex=True
    ).str.lower()

    matches = movie_info_df[titles_clean.str.contains(movie_name.lower())]

    if matches.empty:
        return "Movie not found."

    movie_id = matches.iloc[0]["movie_id"]
    scores = item_similarity_df[movie_id].drop(movie_id)

    top_movies = scores.sort_values(ascending=False).head(top_n)

    result = movie_info_df[movie_info_df.movie_id.isin(top_movies.index)]
    result = result.set_index("movie_id").loc[top_movies.index]
    result["similarity"] = top_movies.values

    return result[["movie title", "similarity"]]

# ================= GRADIO UI =================

with gr.Blocks(title="Movie Recommendation System") as demo:
    gr.Markdown("# 🎬 Movie Recommendation System")
    
    with gr.Tab("Similar Movies (Movie Name)"):
        name = gr.Textbox(label="Movie Name")
        out3 = gr.Dataframe()
        gr.Button("Find Similar").click(movie_name_based, name, out3)

    with gr.Tab("User-Based (User ID)"):
        uid = gr.Number(label="User ID")
        out1 = gr.Dataframe()
        gr.Button("Recommend").click(user_based_recommend, uid, out1)

    with gr.Tab("Item-Based (User ID)"):
        uid2 = gr.Number(label="User ID")
        out2 = gr.Dataframe()
        gr.Button("Recommend").click(item_based_user, uid2, out2)


demo.launch()
