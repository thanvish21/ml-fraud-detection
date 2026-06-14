import React, { useEffect, useState } from 'react';
import {
  RadialBarChart, RadialBar, PolarAngleAxis,
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, BarChart, Bar, Legend,
} from 'recharts';

const POLL_MS = 5000;

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

function Gauge({ rate }) {
  const pct = Math.min(rate * 100, 100);
  const data = [{ name: 'rate', value: pct, fill: pct > 5 ? '#ff5d6c' : pct > 2 ? '#f0b429' : '#4fd1c5' }];
  return (
    <ResponsiveContainer width="100%" height={220}>
      <RadialBarChart cx="50%" cy="50%" innerRadius="65%" outerRadius="100%" data={data} startAngle={180} endAngle={0}>
        <PolarAngleAxis type="number" domain={[0, 100]} dataKey="value" tick={false} />
        <RadialBar dataKey="value" background={{ fill: '#1f2a4a' }} cornerRadius={8} />
        <text x="50%" y="60%" textAnchor="middle" fill="#fff" fontSize="32" fontWeight="600">
          {pct.toFixed(2)}%
        </text>
      </RadialBarChart>
    </ResponsiveContainer>
  );
}

export default function App() {
  const [live, setLive] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [patterns, setPatterns] = useState([]);
  const [modelPerf, setModelPerf] = useState([]);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const [l, t, p, m] = await Promise.all([
          fetchJson('/stats/live'),
          fetchJson('/stats/timeline?minutes=30'),
          fetchJson('/stats/patterns'),
          fetchJson('/stats/model'),
        ]);
        if (cancelled) return;
        setLive(l); setTimeline(t); setPatterns(p); setModelPerf(m); setErr(null);
      } catch (e) {
        if (!cancelled) setErr(e.message);
      }
    }
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const tlData = timeline.map(t => ({
    time: new Date(t.bucket).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    total: t.total, fraud: t.fraud,
  }));

  return (
    <div className="container">
      <h1>Fraud Detection — Live Operations</h1>
      {err && <div className="card" style={{ borderColor: '#ff5d6c', marginBottom: 12 }}>
        <strong>Stream warning:</strong> {err}
      </div>}
      <div className="grid">
        <div className="card">
          <h2>Live Fraud Rate (5m window)</h2>
          <Gauge rate={live?.fraud_rate ?? 0} />
          <div className="sub">
            {live ? `${live.fraud_transactions.toLocaleString()} of ${live.total_transactions.toLocaleString()} flagged · p̄ latency ${live.avg_latency_ms.toFixed(1)} ms` : 'awaiting data…'}
          </div>
        </div>
        <div className="card">
          <h2>Transaction Volume (last 30m)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={tlData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#1f2a4a" />
              <XAxis dataKey="time" stroke="#9aa4c7" />
              <YAxis stroke="#9aa4c7" />
              <Tooltip contentStyle={{ background: '#131a33', border: '1px solid #1f2a4a' }} />
              <Legend />
              <Line type="monotone" dataKey="total" stroke="#4fd1c5" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="fraud" stroke="#ff5d6c" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h2>Top Fraud Patterns (by merchant category, 1h)</h2>
          <table>
            <thead><tr><th>Category</th><th>Total</th><th>Fraud</th><th>Rate</th><th>Avg $</th></tr></thead>
            <tbody>
              {patterns.map(p => (
                <tr key={p.merchant_category}>
                  <td>{p.merchant_category}</td>
                  <td>{p.total.toLocaleString()}</td>
                  <td>{p.fraud.toLocaleString()}</td>
                  <td>{(p.fraud_rate * 100).toFixed(2)}%</td>
                  <td>${p.avg_amount.toFixed(2)}</td>
                </tr>
              ))}
              {patterns.length === 0 && <tr><td colSpan="5" className="sub">no data</td></tr>}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h2>Model Performance (1h)</h2>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={modelPerf} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#1f2a4a" />
              <XAxis dataKey="model_name" stroke="#9aa4c7" />
              <YAxis stroke="#9aa4c7" />
              <Tooltip contentStyle={{ background: '#131a33', border: '1px solid #1f2a4a' }} />
              <Bar dataKey="avg_latency_ms" fill="#4fd1c5" />
            </BarChart>
          </ResponsiveContainer>
          <table>
            <thead><tr><th>Model</th><th>Version</th><th>Requests</th><th>p̄ latency</th><th>p̄ score</th></tr></thead>
            <tbody>
              {modelPerf.map(m => (
                <tr key={m.model_name + m.model_version}>
                  <td>{m.model_name}</td><td>{m.model_version}</td><td>{m.total.toLocaleString()}</td>
                  <td>{m.avg_latency_ms.toFixed(2)} ms</td><td>{m.avg_proba.toFixed(3)}</td>
                </tr>
              ))}
              {modelPerf.length === 0 && <tr><td colSpan="5" className="sub">no data</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
