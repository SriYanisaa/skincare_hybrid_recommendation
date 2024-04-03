from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, Conv1D, BatchNormalization, Dense, GlobalMaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load data
data = pd.read_csv('data_skincare_for_modeling_2_2.csv')

# Preprocess data
skincare_data_unique = data.drop_duplicates(subset=['description_processed'], keep='first')
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(skincare_data_unique['description_processed'] + ' ' + skincare_data_unique['subcategory'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Encode categories
data_category = data['subcategory'].unique().tolist()
category_to_category_encoded = {x: i for i, x in enumerate(data_category)}
category_encoded_to_category = {i: x for i, x in enumerate(data_category)}
data['category_id'] = data['subcategory'].map(category_to_category_encoded)

# Encode skincare products
skincare_ids = data['product_id'].unique().tolist()
skincare_to_skincare_encoded = {x: i for i, x in enumerate(skincare_ids)}
skincare_encoded_to_skincare = {i: x for i, x in enumerate(skincare_ids)}

# Preprocess ratings
data['star_rating'] = data['star_rating'].values.astype(np.float32)
min_rating = min(data['star_rating'])
max_rating = max(data['star_rating'])

# Prepare data for collaborative filtering model
x = data[['category_id', 'product_id']].values
y = data[['star_rating']].values
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)

num_skincare = len(skincare_encoded_to_skincare)
num_categories = len(category_to_category_encoded)

class RecommenderNet(Model):
    def __init__(self, num_categories, num_skincare, embedding_size, cnn_filters, cnn_kernel_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_categories = num_categories
        self.num_products = num_skincare
        self.embedding_size = embedding_size
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size

        self.category_embedding = Embedding(
            num_categories,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer='l2'
        )
        self.product_embedding = Embedding(
            num_skincare,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer='l2'
        )

        self.cnn_layers = []
        for _ in range(2):
            self.cnn_layers.append(Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu'))
        
        self.batch_norm_layers = [BatchNormalization() for _ in range(3)]
        
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')

        self.output_layer = Dense(1, activation='linear')

    def call(self, inputs):
        category_input, product_input = inputs
        category_vector = self.category_embedding(category_input)
        product_vector = self.product_embedding(product_input)

        concatenated = tf.concat([category_vector, product_vector], axis=1)

        for i, cnn_layer in enumerate(self.cnn_layers):
            concatenated = cnn_layer(tf.expand_dims(concatenated, axis=2))
            concatenated = GlobalMaxPooling1D()(concatenated)
            concatenated = self.batch_norm_layers[i](concatenated)
        
        x = self.dense1(concatenated)
        x = self.dense2(x)

        output = self.output_layer(x)

        return output

# Instantiate the model
model = RecommenderNet(num_categories, num_skincare, 50, cnn_filters=32, cnn_kernel_size=3)

# Call the model to initialize it
dummy_category_input = np.array([0])  # Provide dummy input
dummy_product_input = np.array([0])   # Provide dummy input
_ = model([dummy_category_input, dummy_product_input])  # Call the model

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Load weights after compiling the model
model.load_weights('recommender_model_cnn_weights.h5') 

@app.route('/')
def home():
    categories = data['subcategory'].unique().tolist()  # Ambil daftar kategori
    return render_template('index.html', categories=categories)

@app.route('/recommend', methods=['POST'])
def recommend():
    form_data = request.form
    category = form_data['category']
    skin_type = form_data['skin_type']
    used_products = form_data['previous_skincare']
    incompatible_ingredients = form_data['incompatible_ingredients']
    
    input_description = f'{category} {skin_type} {used_products}'

    input_tfidf_matrix = tfidf.transform([input_description])
    input_cosine_sim = cosine_similarity(input_tfidf_matrix, tfidf_matrix)

    category_id = category_to_category_encoded[category]
    products_in_category = data[data['category_id'] == category_id]

    # Get top 5 most similar products based on cosine similarity
    top_n_similar_products_idx = np.argsort(input_cosine_sim.flatten())[::-1][:5]
    top_n_similar_product_ids = products_in_category.iloc[top_n_similar_products_idx]['product_id'].tolist()

    # Collaborative Filtering Score
    category_input_cnn = np.array([category_id])
    cf_scores = []
    for product_id in top_n_similar_product_ids:
        product_input_cnn = np.array([product_id])
        cf_score = model.predict([category_input_cnn, product_input_cnn])
        cf_scores.append(cf_score)

    # Content-Based Filtering Score
    content_scores = []
    for idx, product_id in enumerate(products_in_category['product_id']):
        if idx < tfidf_matrix.shape[0]:  # Ensure idx is within the range of tfidf_matrix
            content_score = cosine_similarity(input_tfidf_matrix, tfidf_matrix[idx])
            content_scores.append(content_score)
        else:
            print(f"Index {idx} is out of range for tfidf_matrix.")

    # Hybrid recommendation (weighted sum)
    alpha = 0.5
    hybrid_scores = alpha * cf_score + (1 - alpha) * np.array(content_scores)

    # Get top N recommendations
    N = int(form_data.get('n', 5))  # Mengambil nilai n dari form, defaultnya 5 jika tidak ada
    if len(products_in_category) < N:
        N = len(products_in_category)  # Sesuaikan N jika produk yang tersedia kurang dari N
    top_n_indices = np.argsort(hybrid_scores.flatten())[::-1][:N]
    top_n_products = [products_in_category.iloc[i]['product_id'] for i in top_n_indices]

    # Convert indices to product names and brands
    top_n_products_info = data.loc[data['product_id'].isin(top_n_products),
                                   ['product_id', 'product_name', 'brand', 'image_url', 'price']]
    top_n_products_info = top_n_products_info.drop_duplicates(subset=['product_id'])  # Remove duplicates
    top_n_products_info = top_n_products_info.values.tolist()

    # Ambil daftar kategori
    categories = data['subcategory'].unique().tolist()

    # Render template index.html dengan data rekomendasi
    return render_template('index.html', categories=categories, recommendations=top_n_products_info)

if __name__ == '__main__':
    app.run(debug=True)