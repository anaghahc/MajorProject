from flask import Flask, request, jsonify
import torch
import numpy as np
from model_loader import load_model
from utils import get_dates, prepare_input_tensor, denormalise, load_model_by_day, run_prediction, station_coords, keys
import pandas as pd
import traceback
from flask_cors import CORS
from collections import deque
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor_cache = {}
recent_dates = deque(maxlen=2) 

# Load model and adjacency support
adj = pd.read_excel('adj.xlsx')
A = torch.tensor(adj.values, dtype=torch.float32).to(device)
supports = [A]  # Replace with your adjacency matrices
aptinit = A
# model = load_model(device, supports, aptinit).to(device)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        content = request.get_json()
        today_date_str = content.get('today_date')
        day = int(content.get('day'))
        feature = content.get('feature')

        # Check if this is a new date
        if today_date_str not in recent_dates:
            recent_dates.append(today_date_str)
            if len(recent_dates) == 2 and all(d in input_tensor_cache for d in recent_dates):
                # More than 2 unique dates used, clear all except the most recent
                to_keep = recent_dates[-1]
                input_tensor_cache.clear()
                recent_dates.clear()
                recent_dates.append(to_keep)

        # Get or compute input tensor
        if today_date_str in input_tensor_cache:
            # input_tensor, show_date_str = input_tensor_cache[today_date_str]
            input_tensor = input_tensor_cache[today_date_str]
        else:
            input_tensor, show_date_str = prepare_input_tensor(today_date_str)
            input_tensor = input_tensor.to(device)
            # input_tensor_cache[today_date_str] = (input_tensor, today_date_str)
            input_tensor_cache[today_date_str] = (input_tensor)

        # Load model
        model = load_model_by_day(day, device, supports, aptinit)
        
        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            denomr = denormalise(output)
            prediction = run_prediction(denomr, day)

        today_date = datetime.strptime(today_date_str, "%Y-%m-%d")
        show_date_str = (today_date + timedelta(days=day)).strftime("%Y-%m-%d")
        print(f'prediction for {show_date_str} when today is {today_date_str}')
        
        return jsonify({
            "prediction": prediction,
            "feature_names": keys[:6],
            "station_coords": station_coords,
            "pred_date": show_date_str
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