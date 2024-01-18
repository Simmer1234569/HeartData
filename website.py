import torch
from flask import Flask, render_template, request
from model import Model
import numpy as np

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.form
    data = np.array([data['age'], data['sex'], data['chol'], data['thalach'], data['tresbps']])
    data = torch.from_numpy(data.astype(np.float32)).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(data)
        print(f"Model output: {output.item()}")  # Print the raw output
        prediction = 'High risk' if output.item() > 0.5 else 'Low risk'

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)