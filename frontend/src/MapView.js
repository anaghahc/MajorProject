import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import L from 'leaflet';

const MapView = ({ data }) => {
  const { prediction, feature_names, station_coords, pred_date } = data;

  const numFeatures = feature_names.length;
  const numStations = Object.keys(station_coords).length;
  const numTimesteps = prediction[0][0].length;

  // Dates
  const dateRange = Array.from({ length: numTimesteps }, (_, i) => {
    const d = new Date(pred_date);
    d.setDate(d.getDate() + i);
    return d.toISOString().split('T')[0];
  });

  // Map centering
  const coords = Object.values(station_coords);
  const center = coords.reduce(
    ([latSum, lonSum], [lat, lon]) => [latSum + lat, lonSum + lon],
    [0, 0]
  ).map(sum => sum / coords.length);

  const referenceFeatureIndex = 0; // PM10
  const refVals = Object.values(station_coords).map((_, i) =>
    prediction[referenceFeatureIndex][i][numTimesteps - 1]
  );
  const minVal = Math.min(...refVals);
  const maxVal = Math.max(...refVals);

  const getColor = (value) => {
    const ratio = (value - minVal) / (maxVal - minVal + 1e-9);
    const blue = 255 - Math.round(ratio * 255);
    return `rgb(${blue}, ${blue}, 255)`; // blue gradient
  };

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
          radius={8}
          color={getColor(refVals[i])}
          fillColor={getColor(refVals[i])}
          fillOpacity={0.8}
        >
          <Popup>
            <strong>{station}</strong><br />
            {feature_names.map((fname, fIdx) => (
              <div key={fname}>
                <b>{fname}:</b><br />
                {prediction[fIdx][i].map((val, t) => (
                  <span key={t}>{dateRange[t]}: {val.toFixed(2)}<br /></span>
                ))}
                <br />
              </div>
            ))}
          </Popup>
        </CircleMarker>
      ))}
    </MapContainer>
  );
};

export default MapView;
