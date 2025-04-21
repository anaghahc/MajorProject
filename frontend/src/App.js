import React, { useState } from 'react';
import 'leaflet/dist/leaflet.css';

import MapView from './MapView';

function App() {
  const [predDate, setPredDate] = useState('');
  const [mapData, setMapData] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pred_date: predDate })
    });
    const data = await res.json();
    setMapData(data);
  };

  return (
    <div>
      <form onSubmit={handleSubmit} style={{ padding: '1rem' }}>
        <label>
          Date:
          <input type="date" value={predDate} onChange={e => setPredDate(e.target.value)} required />
        </label>
        <button type="submit">Submit</button>
      </form>
      {mapData && <MapView data={mapData} />}
    </div>
  );
}

export default App;