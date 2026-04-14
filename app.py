import os
import pickle

if not os.path.exists("model.pkl"):
    import train_model

model = pickle.load(open('model.pkl', 'rb'))
accuracy = pickle.load(open('accuracy.pkl', 'rb'))

from flask import Flask, render_template, request
import pickle
import matplotlib.pyplot as plt

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
accuracy = pickle.load(open('accuracy.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = int(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    location = int(request.form['location'])

    result = model.predict([[area, bedrooms, location]])

    # Graph
    plt.figure(figsize=(4,3))
    plt.bar(['Price'], [result[0]], color='#6fa8dc')
    plt.title('Predicted House Price')
    plt.ylabel('Amount')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('static/graph.png')
    plt.close()

    return render_template(
        'index.html',
        prediction_text=f"Predicted Price: ₹{int(result[0]):,}",
        accuracy=f"Model Accuracy (R² Score): {round(accuracy*100,2)}%",
        graph="static/graph.png"
    )

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=10000)