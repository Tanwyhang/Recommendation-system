import numpy as np
import pandas as pd

def preview_data(var, line_spacing=3) -> None:
    # Search through all variables in the global scope
    for name, value in globals().items():
        # If the value matches the provided variable, we assume we've found it
        if value is var:
            print("\n"*line_spacing, f"{name}:\n\n {value}", "\n"*line_spacing)
            break
    

# Step 1: Create a mock user-song rating matrix with some missing values (np.nan)
# Each row represents a user, and each column represents a song.
# index: user, column: songs
ratings_data = {
    "Song A": [5,       4, np.nan,      1, np.nan],
    "Song B": [3,  np.nan,      2,      5, np.nan],
    "Song C": [4,       3, np.nan,      4,      3],
    "Song D": [np.nan,  2,      3, np.nan,      4],
}
preview_data(ratings_data)


# each rating can be only provided by one user
indices = []
for i in range(len(ratings_data["Song A"])):
    indices.append(f"User {i + 1}")
preview_data(indices)
# indices = ["User 1", "USer 2", ...]


# Convert the data to a pandas DataFrame for easy manipulation and visualization row = user column = songdata
ratings_df = pd.DataFrame(ratings_data, index=indices)
preview_data(ratings_df)

# Step 2: Fill missing values with each user's average rating
# This is a simple imputation technique so that the SVD algorithm can work with a fully populated matrix.
# the parameter axis=1 is used to specify that the mean should be calculated for each ROW (this provides the mean rating given from each user).
# axis=0 (default): Operates vertically down the rows for each column, so the mean is calculated for each column individually. This would provide the average rating for each song across all users.
user_means = ratings_df.mean(axis=1)
ratings_filled = ratings_df.apply(lambda row: row.fillna(user_means[row.name]), axis=1)

# Step 3: Convert the DataFrame to a NumPy array for matrix operations with SVD
R = ratings_filled.values
preview_data(R)

# Latent features are hidden traits of songs and users that we can’t see directly.
# For example, a latent feature might represent whether a song is upbeat or slow.

# Step 4: Perform Singular Value Decomposition (SVD) on the rating matrix
# SVD breaks down the big table of user ratings into three smaller tables:

# - U contains how much each user relates to different hidden traits (like upbeat or slow).
# - Sigma (Σ) contains numbers that show how important each hidden trait is.
# - Vt contains how each song relates to those same hidden traits (like whether a song is more upbeat or mellow).

U, sigma, Vt = np.linalg.svd(R, full_matrices=False)
preview_data(U)
preview_data(sigma)
preview_data(Vt)

# Step 5: Reconstruct the matrix using a reduced rank (k)
# Reducing the rank allows us to capture only the most significant latent features
# Here, we choose k = 2 (number of latent features) for simplicity

'''
High 𝑘 → more details, better accuracy, but slower and risks overfitting.
Low 𝑘 → faster, more generalized, but risks underfitting.
'''

k = 4
sigma = np.diag(sigma[:k])  # Convert the first k singular values into a diagonal matrix
U_k = U[:, :k]              # Take the first k columns of U
Vt_k = Vt[:k, :]            # Take the first k rows of Vt

# Step 6: Calculate the predicted rating matrix by multiplying U_k, sigma, and Vt_k
# The resulting matrix R_pred contains estimated ratings for each user->song pair
R_pred = np.dot(np.dot(U_k, sigma), Vt_k)

# Step 7: Convert the predicted matrix back into a DataFrame for easier interpretation
predicted_ratings_df = pd.DataFrame(R_pred, index=ratings_df.index, columns=ratings_df.columns)
preview_data(predicted_ratings_df)

# Step 8: Generate song recommendations for each user
# For each user, find songs they haven't rated yet (i.e., in the original ratings_df)
# Sort these songs by predicted rating in descending order and recommend the top ones
for user in ratings_df.index:
    user_ratings = ratings_df.loc[user]        # Get the original ratings for this user
    user_predictions = predicted_ratings_df.loc[user]  # Get the predicted ratings

    # Find songs the user hasn't rated/listen yet, and sort by predicted rating
    recommendations = user_predictions[user_ratings.isna()].sort_values(ascending=False)
    # Display the recommendations for this user
    print(f"Recommendations for {user}:")
    for song, rating in recommendations.items():
        print(f"song  : {song}\nrating: {rating}\n")
