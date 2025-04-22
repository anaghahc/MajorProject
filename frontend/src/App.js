import React, { useState } from 'react';
import 'leaflet/dist/leaflet.css';
import './index.css';
import MapView from './MapView';

function App() {
  const [predDate, setPredDate] = useState('');
  const [selectedDay, setSelectedDay] = useState('1');
  const [selectedFeature, setSelectedFeature] = useState('pm10');
  const [mapData, setMapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          today_date: predDate,
          day: parseInt(selectedDay),
          feature: selectedFeature
        })
      });

      if (!res.ok) throw new Error('Failed to fetch prediction.');
      const data = await res.json();
      setMapData(data);
    } catch (err) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const days = Array.from({ length: 31 }, (_, i) => (i + 1).toString());
  const features = ['pm10', 'pm2.5', 'co', 'no2', 'so2', 'o3'];

  return (
    <div className="app-container">
      <form onSubmit={handleSubmit} className="form">
        <label>
          <span>Today's Date:</span>
          <input type="date" value={predDate} onChange={e => setPredDate(e.target.value)} required />
        </label>

        <label>
          <span>Select Day (1-31):</span>
          <select value={selectedDay} onChange={e => setSelectedDay(e.target.value)}>
            {days.map(d => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>

        <label>
          <span>Select Feature:</span>
          <select value={selectedFeature} onChange={e => setSelectedFeature(e.target.value)}>
            {features.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>

        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Submit'}
        </button>

        {error && <div className="error">{error}</div>}
      </form>

      {mapData && <MapView data={mapData} selectedFeature={selectedFeature} />}
    </div>
  );
}

export default App;
