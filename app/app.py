from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib


app = Flask(__name__)

Toxic_model = load_model('model.h5')

max_len = 200 

tokenizer = joblib.load('tokenizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.json['comment']
    
    new_comment_seq = tokenizer.texts_to_sequences([comment])
    new_comment_pad = pad_sequences(new_comment_seq, maxlen=max_len)
    
    predictions = Toxic_model.predict(new_comment_pad)
    predicted_label = label_encoder.inverse_transform([1 if pred > 0.5 else 0 for pred in predictions])
    result = int(predicted_label[0])
    
    return jsonify({'predicted_label': result})
