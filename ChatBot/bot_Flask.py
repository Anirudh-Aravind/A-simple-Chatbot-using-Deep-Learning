
from flask import Flask, render_template, jsonify, request
import processor
import json
import numpy as np
from tensorflow import keras
import pickle

with open("intents.json") as file:
    data = json.load(file)
model = keras.models.load_model('chat_model')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('new.html', **locals())
@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
            the_question = request.form['question']

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([the_question]),
                                                                      truncating='post', maxlen=20))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            response = np.random.choice(i['responses'])
    return jsonify({"response": response })
if __name__ == '__main__':
    app.run(debug=True)