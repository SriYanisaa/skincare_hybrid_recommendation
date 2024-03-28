from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = Flask(__name__)

# Load data
data = pd.read_csv('data_skincare_for_modeling_2_2.csv')
skincare_data_unique = data.drop_duplicates(subset=['description_processed'], keep='first')
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(skincare_data_unique['description_processed'] + ' ' + skincare_data_unique['subcategory'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Collaborative Filtering Model
data_category = data['subcategory'].unique().tolist()
category_to_category_encoded = {x: i for i, x in enumerate(data_category)}
data['category_id'] = data['subcategory'].map(category_to_category_encoded)

skincare_ids = data['product_id'].unique().tolist()
skincare_to_skincare_encoded = {x: i for i, x in enumerate(skincare_ids)}

x = data[['category_id', 'product_id']].values
y = data[['star_rating']].values
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
y_train = scaler.fit_transform(y_train)
y_val = scaler.transform(y_val)

num_skincare = len(skincare_to_skincare_encoded)
num_categories = len(category_to_category_encoded)

class RecommenderNet(tf.keras.Model):
    def __init__(self, num_categories, num_skincare, embedding_size, cnn_filters, cnn_kernel_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_categories = num_categories
        self.num_products = num_skincare
        self.embedding_size = embedding_size
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size

        self.category_embedding = tf.keras.layers.Embedding(
            num_categories,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.product_embedding = tf.keras.layers.Embedding(
            num_skincare,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )

        self.cnn_layers = []
        for _ in range(2):
            self.cnn_layers.append(tf.keras.layers.Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu'))
        
        self.batch_norm_layers = [tf.keras.layers.BatchNormalization() for _ in range(3)]
        
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        category_input, product_input = inputs
        category_vector = self.category_embedding(category_input)
        product_vector = self.product_embedding(product_input)

        concatenated = tf.concat([category_vector, product_vector], axis=1)

        for i, cnn_layer in enumerate(self.cnn_layers):
            concatenated = cnn_layer(tf.expand_dims(concatenated, axis=2))
            concatenated = tf.keras.layers.GlobalMaxPooling1D()(concatenated)
            concatenated = self.batch_norm_layers[i](concatenated)
        
        x = self.dense1(concatenated)
        x = self.dense2(x)
        output = self.output_layer(x)

        return output

model = RecommenderNet(num_categories, num_skincare, 50, cnn_filters=32, cnn_kernel_size=3)
model.load_weights('recommender_model_cnn_weights.h5') 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    if request.method == 'POST':
        category = request.form['category']
        skin_type = request.form['skin_type']
        previous_products = request.form['previous_products']
        problematic_ingredients = request.form['problematic_ingredients']

        # Process the input and get recommendations
        # Your recommendation logic here based on the input

        recommendations = ["Product 1", "Product 2", "Product 3"]  # Dummy recommendations

        return render_template('recommendation.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
