from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model1 = pickle.load(open(r"C:\Users\Madhumitha\PycharmProjects\movie_intrest_webapp\data\model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from HTML form
    val1 = float(request.form['feature1'])
    val2 = float(request.form['feature2'])
    val3 = float(request.form['feature3'])
    val4 = float(request.form['feature4'])

    # Format input data for prediction
    input_data = np.array([[val1, val2], [val3, val4]])
    predictions = model1.predict(input_data).tolist()

    return render_template('result.html', prediction=predictions)

if __name__ == '_main_':
    app.run(debug=True)
