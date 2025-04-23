from datetime import datetime, timedelta
import ee
import pandas as pd
import json
import requests
import  numpy as np
import torch
from model_loader import load_model
import os
from ee import ServiceAccountCredentials

# ee.Authenticate()
HERE = os.path.dirname(__file__)
key_path = os.path.join(HERE, "ee-service-account.json")
credentials = ServiceAccountCredentials(None, key_path)
ee.Initialize(credentials)


# Load JSON from file
with open("normalization_stats.json", "r") as f:
    data = json.load(f)

# Get the top-level keys
keys = list(data.keys())

station_coords = {
    "Central Delhi": (28.6443, 77.2420),
    "East Delhi": (28.6380, 77.2936),
    "New Delhi": (28.632778, 77.219722),
    "North Delhi": (28.658813, 77.216742),
    "North East Delhi": (28.6640, 77.2712),
    "North West Delhi": (28.72674, 77.00248),
    "Shahdara": (28.6733, 77.26768),
    "South Delhi": (28.5192, 77.2134),
    "South East Delhi": (28.5531, 77.2599),
    "South West Delhi": (28.5574, 77.159),
    "West Delhi": (28.644184, 77.11182),
    "Bhiwani": (28.7833, 76.1333),
    "Charkhi Dadri": (28.5921, 76.2653),
    "Faridabad": (28.4211, 77.3078),
    "Gurgaon": (28.456, 77.029),
    "Jhajjar": (28.6055, 76.6555),
    "Jind": (29.3166, 76.3166),
    "Karnal": (29.686, 76.989),
    "Mahendragarh": (28.0444, 76.1083),
    "Nuh": (28.1024, 76.9931),
    "Palwal": (28.1473, 77.3260),
    "Panipat": (29.3909, 76.9635),
    "Rewari": (28.1920, 76.6191),
    "Rohtak": (28.8909, 76.5796),
    "Sonipat": (28.990, 77.022),
    "Alwar": (27.549780, 76.635539),
    "Bharatpur": (27.2173, 77.4901),
    "Baghpat": (28.9428, 77.2276),
    "Bulandshahr": (28.406944, 77.849722),
    "Gautam Buddh Nagar": (28.5333, 77.3891),
    "Ghaziabad": (28.6692, 77.4538),
    "Hapur": (28.730937, 77.775736),
    "Meerut": (28.9845, 77.7064),
    "Muzaffarnagar": (29.4727, 77.7085),
    "Shamli": (29.4502, 77.3172)
}

def load_model_by_day(day, device, supports, aptinit):
    model_day = day if day % 2 == 1 else day + 1
    model_path = f"models/WaveNet_13_{model_day}.pt"
    model = load_model(device, supports, aptinit, model_day).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def run_prediction(pred, day):
    return pred[:,:,day-1:day].cpu().numpy().tolist()


weather_parameters = {
    "temperature": "Temperature",
    "wind_speed_10m": "Wind Speed",
    "surface_pressure": "Surface Pressure"
}

aqi_parameters = {
    "pm10":'pm10', "pm2_5":"pm2.5", "carbon_monoxide":"co", "nitrogen_dioxide":"no2", "sulphur_dioxide":"so2", "ozone":"o3"
}

def get_dates(pred_date_str):

    # Convert the end date string to a datetime object
    pred_date = datetime.strptime(pred_date_str, "%Y-%m-%d")

    end_date = pred_date - timedelta(days=1)
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Calculate the start date by subtracting 13 days
    start_date = pred_date - timedelta(days=13)

    # Convert the start date back to string format
    start_date_str = start_date.strftime("%Y-%m-%d")

    # print("Start Date:", start_date_str)
    # print("\nEnd Date:", end_date_str)

    return start_date_str, end_date_str

def fetch_weather_data(lat, lon, param, start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={param}&timezone=America/Los_Angeles"
    response = requests.get(url)
    data = response.json()

    if "hourly" not in data:
        return None

    times = data["hourly"]["time"]
    values = data["hourly"][param]

    # Extract data at 8 AM (08:00 IST)
    daily_data = []
    for i in range(len(times)):
        dt = datetime.fromisoformat(times[i])
        if dt.hour == 13:  # 1 PM in IST
            daily_data.append((dt.date(), values[i]))

    return daily_data

def fetch_aqi_data(lat, lon, param, start_date, end_date):
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly={param}&timezone=Asia/Kolkata"
    response = requests.get(url)
    data = response.json()

    if "hourly" not in data:
        return None

    times = data["hourly"]["time"]
    values = data["hourly"][param]

    # Extract data at 8 AM (08:00 IST)
    daily_data = []
    for i in range(len(times)):
        dt = datetime.fromisoformat(times[i])
        if dt.hour == 13:  # 1 PM in IST
            daily_data.append((dt.date(), values[i]))

    return daily_data


# Fetch data for one pollutant
def get_pollutant_data(start, end, collection_id, band_name, pollutant_label, district_fc):
    try:
        image = ee.ImageCollection(collection_id) \
            .filterDate(start, end) \
            .filterBounds(district_fc.geometry()) \
            .select(band_name) \
            .mean()

        results = image.reduceRegions(
            collection=district_fc,
            reducer=ee.Reducer.mean(),
            scale=1113
        )

        results_list = results.toList(results.size()).getInfo()

        return [{
            "Date": start,
            "District": f["properties"]["district"],
            pollutant_label: f["properties"].get("mean", None)
        } for f in results_list]
    except Exception as e:
        print(f"Error fetching {pollutant_label} on {start}: {e}")
        return []



def prepare_input_tensor(today_date_str):
    # Convert to datetime object
    today_date = datetime.strptime(today_date_str, "%Y-%m-%d")

    # Add one day
    pred_day = today_date + timedelta(days=1)

    # Convert back to string
    pred_date_str = pred_day.strftime("%Y-%m-%d")
    
    # Dictionary to store data for all parameters
    weather_data = {param: [] for param in weather_parameters.keys()}
    start_date, end_date = get_dates(pred_date_str)

    # Fetch data for all districts and parameters
    for district, (lat, lon) in station_coords.items():
        # print(f"Fetching data for {district}...")
        for param in weather_parameters.keys():
            print(f"  - {weather_parameters[param]}")
            data = fetch_weather_data(lat, lon, param, start_date, end_date)
            if data:
                for date, value in data:
                    weather_data[param].append([district, date, value])


    weather_dfs = {}

    # Loop through the weather_data and create DataFrames
    for param, data in weather_data.items():
        df = pd.DataFrame(data, columns=["District", "Date", weather_parameters[param]])
        weather_dfs[weather_parameters[param]] = df


    
    # Dictionary to store data for all parameters
    aqi_data = {param: [] for param in aqi_parameters.keys()}
    start_date, end_date = get_dates(pred_date_str)

    # Fetch data for all districts and parameters
    for district, (lat, lon) in station_coords.items():
        # print(f"Fetching data for {district}...")
        for param in aqi_parameters.keys():
            print(f"  - {aqi_parameters[param]}")
            data = fetch_aqi_data(lat, lon, param, start_date, end_date)
            if data:
                for date, value in data:
                    aqi_data[param].append([district, date, value])

    aqi_dfs = {}

    # Loop through the weather_data and create DataFrames
    for param, data in aqi_data.items():
        df = pd.DataFrame(data, columns=["District", "Date", aqi_parameters[param]])
        aqi_dfs[aqi_parameters[param]] = df 

    

    # Your district coordinates dictionary
    district_features = [ee.Feature(ee.Geometry.Point(lon, lat), {"district": district})
                        for district, (lat, lon) in station_coords.items()]
    district_fc = ee.FeatureCollection(district_features)

    # Pollutants to fetch
    collections = {
        "Cloud": ("COPERNICUS/S5P/NRTI/L3_CLOUD", "cloud_fraction"),
        "CO": ("COPERNICUS/S5P/NRTI/L3_CO", "CO_column_number_density"),
        "O3": ("COPERNICUS/S5P/NRTI/L3_O3", "O3_column_number_density")
    }

    # 2-day time windows
    start_date_sat = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date_sat = datetime.strptime(pred_date_str, "%Y-%m-%d").date() + timedelta(days=1)
    delta = timedelta(days=1)
    date_ranges = [(start_date_sat + i * delta).strftime("%Y-%m-%d") for i in range((end_date_sat - start_date_sat) // delta)]

    # Collect all pollutants per date, then merge
    all_records = []

    for i in range(len(date_ranges) - 1):
        start, end = date_ranges[i], date_ranges[i + 1]

        merged = {}
        for pollutant, (collection_id, band_name) in collections.items():
            data = get_pollutant_data(start, end, collection_id, band_name, pollutant, district_fc)

            for entry in data:
                key = (entry["District"], entry["Date"])
                if key not in merged:
                    merged[key] = {"District": entry["District"], "Date": entry["Date"]}
                merged[key][pollutant] = entry.get(pollutant)

        all_records.extend(merged.values())

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Sort and reorder columns
    df = df[["District", "Date", "Cloud", "CO", "O3"]]
    df.sort_values(["District", "Date"], inplace=True)

    # Load your wide-format DataFrame (if not already loaded)
    # df = pd.read_csv("satellite_data/pollutants_wide.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Define full date range
    full_dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Empty list to store filled data per district
    filled_data = []

    # Fill missing data for each district separately
    for district in df["District"].unique():
        sub_df = df[df["District"] == district].set_index("Date")

        # Reindex to full date range
        sub_df = sub_df.reindex(full_dates)

        # Add back District column
        sub_df["District"] = district

        # Interpolate and fill missing
        sub_df = sub_df.interpolate(method="linear").ffill().bfill()

        # Reset index and rename
        sub_df.reset_index(inplace=True)
        sub_df.rename(columns={"index": "Date"}, inplace=True)

        filled_data.append(sub_df)

    # Combine all filled data
    df_filled = pd.concat(filled_data, ignore_index=True)


    for df in list(aqi_dfs.values()) + list(weather_dfs.values()):
        df["Date"] = pd.to_datetime(df["Date"])
    df_filled["Date"] = pd.to_datetime(df_filled["Date"])

    # Step 1: Merge weather DataFrames
    weather_merged = None
    for feature, df in weather_dfs.items():
        if weather_merged is None:
            weather_merged = df
        else:
            weather_merged = pd.merge(weather_merged, df, on=["District", "Date"], how="outer")

    # Step 2: Merge AQI DataFrames
    aqi_merged = None
    for feature, df in aqi_dfs.items():
        if aqi_merged is None:
            aqi_merged = df
        else:
            aqi_merged = pd.merge(aqi_merged, df, on=["District", "Date"], how="outer")

    # Step 3: Merge weather and AQI together
    final_df = pd.merge(aqi_merged, weather_merged, on=["District", "Date"], how="outer")

    # Step 4: Merge with satellite data
    final_df = pd.merge(final_df, df_filled, on=["District", "Date"], how="outer")

    # Step 5: Sort and reset index
    final_df = final_df.sort_values(by=["District", "Date"]).reset_index(drop=True)

    with open("normalization_stats.json", "r") as f:
        data = json.load(f)

    # Get the top-level keys
    keys = list(data.keys())

    final_df.columns = list(final_df.columns[:2]) + keys

    for col, params in data.items():
        if col in final_df.columns:
            col_mean = params["mean"]
            col_std = params["std"]
            final_df[col] = (final_df[col] - col_mean) / col_std if col_std != 0 else final_df[col]

    # Ensure 'Date' is in datetime format
    df = final_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    # Pivot the DataFrame to have (District, Date) as rows and pollutants as columns
    pivot_df = df.pivot_table(index=['District', 'Date'])

    # Get unique districts, pollutants, and dates
    districts = pivot_df.index.get_level_values('District').unique()
    pollutants = pivot_df.columns
    dates = pivot_df.index.get_level_values('Date').unique()

    # Create an empty 3D array of shape (n, m, t)
    n = len(districts)  # Number of districts
    m = len(pollutants)  # Number of pollutants
    t = len(dates)  # Number of days
    data_3d = np.empty((n, m, t))

    # Fill the 3D array
    for i, district in enumerate(districts):
        for j, pollutant in enumerate(pollutants):
            for k, date in enumerate(dates):
                try:
                    data_3d[i, j, k] = pivot_df.loc[(district, date), pollutant]
                except KeyError:
                    data_3d[i, j, k] = np.nan  # Handle missing values

    data_3d_tensor = torch.tensor(data_3d, dtype=torch.float32)  # now shape: [35, 12, 13]

    # Step 2: Slice to keep only first 3 features (from dim=2)
    data_3d_tensor = data_3d_tensor[:, :, :3]  # shape: [35, 12, 3]

    # Step 3: Permute to [1, 12, 35, 3]
    input_tensor = data_3d_tensor.permute(1, 0, 2).unsqueeze(0)


    return input_tensor, pred_date_str


def denormalise(output):
    output = output.squeeze(0)  # if batch_size = 1
    with open('normalization_stats.json', 'r') as f:
        data = json.load(f)

    features = list(data.keys())

    for i, feature in enumerate(features):
        mean = data[feature]["mean"]
        std = data[feature]["std"]
        output[i] = output[i] * std + mean
    return output