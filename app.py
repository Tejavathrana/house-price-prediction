from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model
if not os.path.exists("model.pkl"):
    import train_model

model = pickle.load(open('model.pkl', 'rb'))
accuracy = pickle.load(open('accuracy.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = request.form.get('area')
        bedrooms = request.form.get('bedrooms')
        location = request.form.get('location')

        # Validation
        if not area or not bedrooms:
            return render_template('index.html', result="Please enter all fields")

        area = float(area)
        bedrooms = int(bedrooms)
        location = int(location)

        # Prediction
        result = model.predict([[area, bedrooms, location]])[0]

        # Graph
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,3))
        plt.bar(['Price'], [result])
        plt.title('Predicted House Price')
        plt.tight_layout()
        plt.savefig('static/graph.png')
        plt.close()

        return render_template('index.html',
                               result=round(result, 2),
                               accuracy=round(accuracy * 100, 2))

    except Exception as e:
        return render_template('index.html', result=str(e))


if __name__ == "__main__":
    app.run(debug=True)