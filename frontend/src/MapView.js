import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, DivIcon } from 'react-leaflet';
import L from 'leaflet';

const MapView = ({ data, selectedFeature }) => {
  if (!data || !selectedFeature) return <div>Loading data or no feature selected...</div>;

  const { prediction = [], feature_names = [], station_coords = {}, pred_date } = data;
  const featureIndex = feature_names.indexOf(selectedFeature);
  if (featureIndex === -1) return <div>Selected feature not found in data</div>;

  const numTimesteps = prediction[0]?.[0]?.length || 0;
  if (numTimesteps === 0) return <div>No prediction data available</div>;

  const formatDate = (d) => {
    const day = String(d.getDate()).padStart(2, '0');
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const year = d.getFullYear();
    return `${day}-${month}-${year}`;
  };

  const dateRange = Array.from({ length: numTimesteps }, (_, i) => {
    const d = new Date(pred_date);
    d.setDate(d.getDate() + i);
    return formatDate(d);
  });

  const coords = Object.values(station_coords);
  if (coords.length === 0) return <div>No station coordinates available</div>;

  const center = coords.reduce(
    ([latSum, lonSum], [lat, lon]) => [latSum + lat, lonSum + lon],
    [0, 0]
  ).map(sum => sum / coords.length);

  const colorPalette = [
    '#FF5733', '#33C1FF', '#FF33F6', '#33FF57', '#FFB733', '#7D33FF', '#FF3380',
    '#33FFF6', '#F633FF', '#33FFB7', '#B733FF', '#FFA833', '#33A1FF', '#FF3365',
  ];

  return (
    <MapContainer center={center} zoom={9} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {Object.entries(station_coords).map(([station, [lat, lon]], i) => {
        const value = prediction[featureIndex]?.[i]?.[0] || 0;
        const color = colorPalette[i % colorPalette.length];

        const icon = L.divIcon({
          className: 'custom-pin',
          html: `<div style="
            background-color: rgb(19, 135, 193);
            width: 18px;
            height: 18px;
            border: 2px solid black;
            border-radius: 50% 50% 50% 0;
            transform: rotate(-45deg);
            box-shadow: 0 0 4px rgba(0,0,0,0.4);
            position: relative;
            top: -9px;
          "></div>`
        });
        
        

        return (
          <Marker key={station} position={[lat, lon]} icon={icon}>
            <Popup>
              <div style={{ fontFamily: 'Arial', minWidth: 180 }}>
                <h4 style={{ marginBottom: 4, color: '#2c3e50' }}>{station}</h4>
                <div style={{ fontSize: '0.9em', color: '#444' }}>
                  {dateRange.map((date, idx) => (
                    <div key={idx} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>{date}</span>
                      <span style={{ fontWeight: 600 }}>
                        {prediction[featureIndex]?.[i]?.[idx]?.toFixed(2) ?? 'N/A'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </Popup>
          </Marker>
        );
      })}
    </MapContainer>
  );
};

export default MapView;
