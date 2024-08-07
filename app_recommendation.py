from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras.models import Model
from keras.layers import Embedding, Conv1D, BatchNormalization, Dense, GlobalMaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from flask_caching import Cache
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import time

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load data
data = pd.read_csv('data_skincare_for_modeling_2_2.csv')

progress = 0 

# Preprocess data
nltk.download('stopwords')
nltk.download('punkt')
# Define text preprocessing methods
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F000-\U0001F6FF"  # emoticons & transportasi
        u"\U0001F780-\U0001F7FF"  # simbol, tanda & bendera
        u"\U0001F800-\U0001F8FF"  # emoji surat / katakana
        u"\U0001F900-\U0001F9FF"  # emoji berbagai jenis
        u"\U00002600-\U000026FF"  # simbol matahari & bulan
        u"\U00002700-\U000027BF"  # simbol koin, alat musik, dll.
        u"\U0001F300-\U0001F5FF"  # simbol & markah
        u"\U0001F680-\U0001F6FF"  # transportasi & simbol tempat
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_text(text):
   # Remove emoji
    text = remove_emoji(text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Case folding
    text = text.lower()
    # Tokenization
    words = nltk.word_tokenize(text)
    # Filtering
    filtered_words = [word for word in words if re.match(r'[a-zA-Z]+', word) and not word in stop_words]
    # Stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    # Joining the stemmed words back into a single string
    preprocessed_text = " ".join(stemmed_words)
    return preprocessed_text

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
dummy_category_input = np.array([0]) 
dummy_product_input = np.array([0]) 
_ = model([dummy_category_input, dummy_product_input])  # Call the model

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Load weights after compiling the model
model.load_weights('model_weights_checkpoint.h5') 

@app.route('/')
@cache.cached(timeout=50)
def home():
    categories = data['subcategory'].unique().tolist() 
    return render_template('index.html', categories=categories)

@app.route('/recommend', methods=['POST'])
def recommend():
    global progress
    progress = 0
    form_data = request.form
    category = form_data['category']
    skin_type = form_data['skin_type']
    used_products = form_data['previous_skincare']
    incompatible_ingredients = form_data['incompatible_ingredients']

    incompatible_ingredients = [ingredient.strip() for ingredient in incompatible_ingredients.split(',')]
    
    input_description = f'{category} {skin_type} {used_products}'
    input_description = preprocess_text(input_description)

    input_tfidf_matrix = tfidf.transform([input_description])
    input_cosine_sim = cosine_similarity(input_tfidf_matrix, tfidf_matrix)

    category_id = list(category_to_category_encoded.keys()).index(category)
    top_n_similar_products_idx = np.argsort(input_cosine_sim.flatten())[::-1]
    top_n_similar_products = skincare_data_unique.iloc[top_n_similar_products_idx]
    top_n_similar_products_in_category = top_n_similar_products[top_n_similar_products['subcategory'] == category]

    # Filter out products with incompatible ingredients
    top_n_similar_products_in_category = top_n_similar_products_in_category[~top_n_similar_products_in_category['description'].str.contains('|'.join(incompatible_ingredients), case=False)]
    top_n_similar_product_ids = top_n_similar_products_in_category['product_id'].tolist()

    # Collaborative Filtering Score
    category_input_cnn = np.array([category_id])
    cf_scores = []
    for idx, product_id in enumerate(top_n_similar_product_ids):
        product_input_cnn = np.array([product_id])
        cf_score = model.predict([category_input_cnn, product_input_cnn])
        cf_scores.append(cf_score)
        progress = int(((idx + 1) / len(top_n_similar_product_ids)) * 50)
        print(f"CF Progress: {progress}%")  # Cetak kemajuan Collaborative Filtering
        time.sleep(0.1)

    # Content-Based Filtering Score
    content_scores = []
    for idx, product_id in enumerate(top_n_similar_product_ids):
        if idx < tfidf_matrix.shape[0]:  # Ensure idx is within the range of tfidf_matrix
            content_score = cosine_similarity(input_tfidf_matrix, tfidf_matrix[idx])
            content_scores.append(content_score)
            progress = int(((idx + 1) / len(top_n_similar_product_ids)) * 50) + 50
            print(f"Content-Based Progress: {progress}%")  # Cetak kemajuan Content-Based Filtering
            time.sleep(0.1)
        else:
            print(f"Index {idx} is out of range for tfidf_matrix.")

    # Hybrid recommendation (weighted sum)
    alpha = 0.8
    hybrid_scores = []
    for cf_score, content_score in zip(cf_scores, content_scores):
        hybrid_score = alpha * cf_score + (1 - alpha) * content_score
        hybrid_scores.append(hybrid_score[0][0])
    
    sorted_scores_indices = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in sorted_scores_indices]
    sorted_scores = [x[1] for x in sorted_scores_indices]
    sorted_product_ids = [top_n_similar_product_ids[i] for i in sorted_indices]

     # Print hybrid scores
    print("Hybrid Scores (Descending Order):")
    for idx, (product_id, score) in enumerate(zip(sorted_product_ids, sorted_scores)):
        print(f"Rank {idx+1}: Product ID: {product_id}, Hybrid Score: {score}")

    top_n_products_info = []
    for idx, product_id in enumerate(sorted_product_ids):
        product_info = data.loc[(data['product_id'] == product_id) & (data['subcategory'] == category),
                                ['product_id', 'product_name', 'brand', 'image_url', 'price', 'description']]
        product_info = product_info.values.tolist()[0]
        top_n_products_info.append(product_info)

    categories = data['subcategory'].unique().tolist()

    return render_template('index.html', categories=categories, recommendations=top_n_products_info, progress=progress)

@app.route('/progress')
def get_progress():
    global progress
    return jsonify(progress=progress)

if __name__ == '__main__':
    app.run(debug=True)