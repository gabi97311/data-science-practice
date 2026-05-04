import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

def generate_netflix_dataset(n_samples=500):
    types = ['Movie', 'TV Show']
    titles = [f'Content_{i}' for i in range(n_samples)]
    directors = ['Director A', 'Director B', 'Director C', np.nan]
    countries = ['USA', 'India', 'UK', 'South Korea', 'Kazakhstan', np.nan]
    
    ratings = ['PG-13', 'TV-MA', 'PG', 'TV-14', 'R']
    genres = ['Documentaries', 'Dramas', 'Comedies', 'Horror Movies', 'International TV Shows']
    generated_types = np.random.choice(types, n_samples)
    
    data = {
        'show_id': [f's{i}' for i in range(1, n_samples + 1)],
        'type': generated_types,
        'title': titles,
        'director': np.random.choice(directors, n_samples),
        'country': np.random.choice(countries, n_samples),
        'release_year': np.random.randint(1990, 2025, n_samples),
        'rating': np.random.choice(ratings, n_samples),
        'duration': [
            f'{np.random.randint(60, 180)} min' if t == 'Movie' else f'{np.random.randint(1, 5)} Seasons' 
            for t in generated_types
        ],
        'listed_in': np.random.choice(genres, n_samples),
        'description': [
            "A fascinating story about technology and future.",
            "A deep dive into historical events and people.",
            "A lighthearted comedy for the whole family.",
            "Dark secrets revealed in this intense thriller.",
            "Exploring the beauty of nature and wildlife."
        ] * (n_samples // 5)
    }

    df = pd.DataFrame(data)
    
    
    for col in ['director', 'country', 'rating']:
        df.loc[df.sample(frac=0.1).index, col] = np.nan
        
    return df

df = generate_netflix_dataset(1000)

df['director'] = df['director'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Unknown') 

df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='release_year', hue='type', multiple='stack', bins=30, palette='Set2')
plt.title('Content distribution by release year and type')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.7)
plt.show() 


print("-----Decision Tree Classifier------")

le_rating = LabelEncoder()
df['rating_encoded'] = le_rating.fit_transform(df['rating'])
X = df[['release_year', 'rating_encoded', 'duration_num']]
le_type = LabelEncoder()
y = le_type.fit_transform(df['type']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel: Decision Tree")
print(f"Accuracy: {accuracy:.4f}")


print("-----KMeans-----")

X_genres = pd.get_dummies(df['listed_in'])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_genres)

cluster_0_samples = df[df['cluster'] == 0][['title', 'listed_in', 'cluster']].head(5)
print(cluster_0_samples)
