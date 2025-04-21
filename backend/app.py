from flask import Flask, request, jsonify
import torch
import numpy as np
from model_loader import load_model
from utils import get_dates, prepare_input_tensor, denormalise, station_coords, keys
import pandas as pd
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and adjacency support
adj = pd.read_excel('adj.xlsx')
A = torch.tensor(adj.values, dtype=torch.float32).to(device)
supports = [A]  # Replace with your adjacency matrices
aptinit = A
model = load_model(device, supports, aptinit).to(device)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        pred_date_str = data['pred_date']  # e.g., "2025-04-21"

        # 1. Get date range
        start_date_str, end_date_str = get_dates(pred_date_str)

        # 2. Load/generate input data using your preprocessing method
        # For example:
        input_tensor = prepare_input_tensor(pred_date_str)
        input_tensor = input_tensor.to(device)

        # 3. Predict
        with torch.no_grad():
            output = model(input_tensor)
            denomr = denormalise(output)
            prediction = denomr.cpu().numpy().tolist()

        return jsonify({
            "prediction": prediction,
            "feature_names": keys[:6],
            "station_coords": station_coords,
            "pred_date": pred_date_str
            })
    
    except Exception as e:
        error_type = type(e).__name__
        tb = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "type": error_type,
            "traceback": tb
        }), 500

if __name__ == "__main__":
    app.run(debug=True)