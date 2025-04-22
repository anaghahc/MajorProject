import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';

const MapView = ({ data, selectedFeature }) => {
  console.log("Full data object:", data);
  console.log("prediction[0]?.[0]:", data?.prediction?.[0]?.[0]);
  console.log("feature_names:", data?.feature_names);
  console.log("selectedFeature:", selectedFeature);
  console.log("featureIndex:", data?.feature_names?.indexOf(selectedFeature));
  console.log("station_coords:", data?.station_coords);
  // Add data validation
  if (!data || !selectedFeature) {
    return <div>Loading data or no feature selected...</div>;
  }

  const { prediction = [], feature_names = [], station_coords = {}, pred_date } = data;

  // Check if selectedFeature exists
  const featureIndex = feature_names.indexOf(selectedFeature);
  if (featureIndex === -1) {
    return <div>Selected feature not found in data</div>;
  }

  // Safely get numTimesteps
  const numTimesteps = prediction[0]?.[0]?.length || 0;
  if (numTimesteps === 0) {
    return <div>No prediction data available</div>;
  }

  // Generate date range
  const dateRange = Array.from({ length: numTimesteps }, (_, i) => {
    const d = new Date(pred_date);
    d.setDate(d.getDate() + i);
    return d.toISOString().split('T')[0];
  });

  // Safely get reference values
  const refVals = Object.values(station_coords).map((_, i) =>
    prediction[featureIndex]?.[i]?.[0] || 0
  );

  const minVal = Math.min(...refVals);
  const maxVal = Math.max(...refVals);

  const getColor = (value) => {
    const ratio = (value - minVal) / (maxVal - minVal + 1e-9);
    const blue = 255 - Math.round(ratio * 255);
    return `rgb(${blue}, ${blue}, 255)`;
  };

  const coords = Object.values(station_coords);
  if (coords.length === 0) {
    return <div>No station coordinates available</div>;
  }

  const center = coords.reduce(
    ([latSum, lonSum], [lat, lon]) => [latSum + lat, lonSum + lon], 
    [0, 0]
  ).map(sum => sum / coords.length);

  return (
    <MapContainer center={center} zoom={9} style={{ height: '80vh', width: '100%' }}>
      <TileLayer
        attribution='&copy; OpenStreetMap'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {Object.entries(station_coords).map(([station, [lat, lon]], i) => (
        <CircleMarker
          key={station}
          center={[lat, lon]}
          radius={10}
          color={getColor(refVals[i])}
          fillColor={getColor(refVals[i])}
          fillOpacity={0.9}
        >
          <Popup>
            <strong>{station}</strong><br />
            {dateRange.map((date, idx) => (
              <div key={idx}>
                {date}: {prediction[featureIndex]?.[i]?.[idx]?.toFixed(2) || 'N/A'}
              </div>
            ))}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
};

export default MapView;