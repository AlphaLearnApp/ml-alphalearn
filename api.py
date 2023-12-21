from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle
from utils import get_random_question
from ast import literal_eval
import os

app = Flask(__name__)

# Load the model
model = load_model('model_dense.h5')

# Load the data
df = pd.read_csv('Data2.csv')

df['Choices'] = df['Choices'].apply(literal_eval)
df['CombinedChoices'] = df['Choices'].apply(lambda x: ' '.join(x))

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['CombinedChoices'])

joblib.dump(vectorizer, 'vectorizer.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load the random question function
with open('random_question_model.pkl', 'wb') as f:
    pickle.dump(get_random_question, f)

with open('random_question_model.pkl', 'rb') as f:
    get_random_question = pickle.load(f)


correct_answer = None
question = None

@app.route('/', methods=['GET'])
def helloWorld():
    return 'API ONLINE'

@app.route('/random-question', methods=['GET'])
def random_question():
    global correct_answer
    global question
    question, choices, correct_answer = get_random_question(df)
    return jsonify({'question': question, 'choices': choices})

@app.route('/predict', methods=['POST'])
def predict():
    global correct_answer
    global question
    data = request.get_json(force=True)
    user_answer = data['answer']

    if user_answer == correct_answer:
        return jsonify({
            "status": {
                "code": 200,
                "message": "Request successful"
            },
            "result": "Your answer is correct!",
        }), 200
    else:
        user_input_vectorized = vectorizer.transform([question])
        user_input_reordered = tf.sparse.reorder(tf.SparseTensor(
            indices=np.vstack((user_input_vectorized.tocoo().row, user_input_vectorized.tocoo().col)).T,
            values=user_input_vectorized.tocoo().data,
            dense_shape=user_input_vectorized.tocoo().shape
        ))
        predicted_similarity = model.predict(user_input_reordered)

        similar_questions_indices = np.argsort(predicted_similarity[:, 0])[::-1]
        top_question_index = similar_questions_indices[0]

        recommended_video_link = df.loc[top_question_index, 'VidioLink']

        return jsonify({
            "status": {
                "code": 200,
                "message": "Request successful"
            },
            "result": 'Your answer is incorrect.',
            'recommended_video_link': recommended_video_link
        }), 200

if __name__ == '__main__':
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
