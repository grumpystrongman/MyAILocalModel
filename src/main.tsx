import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke } from '@tauri-apps/api/core';
import './styles.css';

type HardwareProfile = {
  os_name: string;
  arch: string;
  cpu_cores: number;
  total_ram_gb: number;
  available_ram_gb: number;
  disk_free_gb: number;
  gpu_name?: string | null;
  vram_gb?: number | null;
  notes: string[];
};

type Recommendation = {
  repo_id: string;
  filename: string;
  quantization?: string | null;
  parameter_billions?: number | null;
  size_gb?: number | null;
  score: number;
  fit_score: number;
  quality_score: number;
  speed_score: number;
  safety_margin_score: number;
  popularity_score: number;
  usability_score: number;
  estimated_memory_gb: number;
  expected_runtime: string;
  runtime_recommendations: string[];
  decision: string;
  reasons: string[];
  cautions: string[];
  download_url: string;
};

type DownloadResult = { path: string; bytes: number };

const tasks = [
  { id: 'general', label: 'General chat', hint: 'Balanced model for normal questions and summaries.' },
  { id: 'coding', label: 'Coding', hint: 'Prioritizes coder/instruct models.' },
  { id: 'writing', label: 'Novel writing', hint: 'Prioritizes expressive instruction models.' },
  { id: 'research', label: 'Research', hint: 'Prioritizes reasoning/context quality.' },
  { id: 'small-business', label: 'Small business private AI', hint: 'Prioritizes easy, safe, local-first models.' },
];

function pct(value: number) {
  return `${Math.round(value * 100)}%`;
}

function explainQuant(q?: string | null) {
  if (!q) return 'Unknown quantization: the app cannot confidently estimate the quality/memory tradeoff from this filename.';
  if (q.startsWith('Q4')) return `${q} is usually the practical sweet spot: strong memory savings, good speed, and acceptable quality loss for most local use.`;
  if (q.startsWith('Q5') || q.startsWith('Q6')) return `${q} keeps more quality than Q4 but uses more memory. Good when your machine has headroom.`;
  if (q.startsWith('Q8')) return `${q} is higher fidelity but heavy. Choose it only when RAM/VRAM is comfortable and speed matters less.`;
  if (q.startsWith('Q2') || q.startsWith('Q3')) return `${q} is very small and fast, but quality can degrade. Useful for constrained machines.`;
  return `${q} has a model-specific tradeoff. Review memory and fit before downloading.`;
}

function App() {
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [task, setTask] = useState('general');
  const [preference, setPreference] = useState(0.5);
  const [status, setStatus] = useState('Ready');
  const [query, setQuery] = useState('GGUF instruct');
  const [downloading, setDownloading] = useState<string | null>(null);

  async function scan() {
    setStatus('Scanning hardware...');
    const result = await invoke<HardwareProfile>('scan_hardware');
    setHardware(result);
    setStatus('Hardware scan complete.');
  }

  async function recommend() {
    setStatus('Searching Hugging Face and scoring models...');
    const result = await invoke<Recommendation[]>('recommend_models', {
      request: { task, query, speed_quality_preference: preference, limit: 12 },
    });
    setRecommendations(result);
    setStatus(`Found ${result.length} recommendations.`);
  }

  async function download(rec: Recommendation) {
    setDownloading(`${rec.repo_id}/${rec.filename}`);
    setStatus(`Downloading ${rec.filename}...`);
    try {
      const result = await invoke<DownloadResult>('download_model', {
        request: { repo_id: rec.repo_id, filename: rec.filename, url: rec.download_url },
      });
      setStatus(`Downloaded ${Math.round(result.bytes / 1024 / 1024)} MB to ${result.path}`);
    } finally {
      setDownloading(null);
    }
  }

  useEffect(() => { scan().catch((err) => setStatus(String(err))); }, []);

  const top = recommendations[0];
  const selectedTask = useMemo(() => tasks.find((t) => t.id === task), [task]);

  return <main>
    <section className="hero">
      <div>
        <p className="eyebrow">Private local AI, without terminal pain</p>
        <h1>MyAILocalModel</h1>
        <p className="lede">Scan your machine, pull real Hugging Face candidates, and explain why one local model is the right choice over another.</p>
      </div>
      <button onClick={() => scan().catch((err) => setStatus(String(err)))}>Rescan hardware</button>
    </section>

    <section className="grid two">
      <div className="card">
        <h2>Hardware intelligence</h2>
        {hardware ? <div className="specs">
          <span>OS</span><strong>{hardware.os_name} {hardware.arch}</strong>
          <span>CPU</span><strong>{hardware.cpu_cores} cores</strong>
          <span>RAM</span><strong>{hardware.available_ram_gb.toFixed(1)} GB free / {hardware.total_ram_gb.toFixed(1)} GB total</strong>
          <span>Disk</span><strong>{hardware.disk_free_gb.toFixed(1)} GB free</strong>
          <span>GPU</span><strong>{hardware.gpu_name || 'Not detected'}</strong>
          <span>VRAM</span><strong>{hardware.vram_gb ? `${hardware.vram_gb.toFixed(1)} GB` : 'Unknown'}</strong>
        </div> : <p>Scanning...</p>}
        {hardware?.notes.map((note) => <p className="note" key={note}>{note}</p>)}
      </div>

      <div className="card controls">
        <h2>Tell me what you're doing</h2>
        <div className="task-list">{tasks.map((t) => <button className={task === t.id ? 'active' : ''} onClick={() => setTask(t.id)} key={t.id}>{t.label}</button>)}</div>
        <p className="note">{selectedTask?.hint}</p>
        <label>Hugging Face query<input value={query} onChange={(e) => setQuery(e.target.value)} /></label>
        <label>Speed ↔ Quality <input type="range" min="0" max="1" step="0.05" value={preference} onChange={(e) => setPreference(Number(e.target.value))} /></label>
        <div className="slider-labels"><span>Faster</span><span>Balanced</span><span>Higher quality</span></div>
        <button className="primary" onClick={() => recommend().catch((err) => setStatus(String(err)))}>Find my best local models</button>
      </div>
    </section>

    <section className="status">{status}</section>

    {top && <section className="card recommendation">
      <p className="eyebrow">Recommended</p>
      <h2>{top.repo_id}</h2>
      <p className="file">{top.filename}</p>
      <p className="decision">{top.decision}</p>
      <div className="bars">
        <label>Fit <span>{pct(top.fit_score)}</span><meter min="0" max="1" value={top.fit_score}/></label>
        <label>Quality <span>{pct(top.quality_score)}</span><meter min="0" max="1" value={top.quality_score}/></label>
        <label>Speed <span>{pct(top.speed_score)}</span><meter min="0" max="1" value={top.speed_score}/></label>
        <label>Safety margin <span>{pct(top.safety_margin_score)}</span><meter min="0" max="1" value={top.safety_margin_score}/></label>
      </div>
      <div className="grid two compact">
        <div><h3>Why this model</h3>{top.reasons.map((r) => <p key={r}>✓ {r}</p>)}</div>
        <div><h3>Cautions</h3>{top.cautions.length ? top.cautions.map((c) => <p key={c}>⚠ {c}</p>) : <p>No major cautions for this hardware profile.</p>}</div>
      </div>
      <p><strong>Quantization:</strong> {explainQuant(top.quantization)}</p>
      <p><strong>Runtime:</strong> {top.expected_runtime}. Also consider {top.runtime_recommendations.join(', ')}.</p>
      <button className="primary" disabled={!!downloading} onClick={() => download(top).catch((err) => setStatus(String(err)))}>{downloading ? 'Downloading...' : 'Download recommended model'}</button>
    </section>}

    <section className="grid cards">
      {recommendations.slice(1).map((rec) => <article className="card model" key={`${rec.repo_id}/${rec.filename}`}>
        <h3>{rec.repo_id}</h3>
        <p className="file">{rec.filename}</p>
        <p>{rec.decision}</p>
        <p>Score {pct(rec.score)} · Fit {pct(rec.fit_score)} · Quality {pct(rec.quality_score)} · Speed {pct(rec.speed_score)}</p>
        <p>Memory estimate: {rec.estimated_memory_gb.toFixed(1)} GB · Quant: {rec.quantization || 'unknown'} · Params: {rec.parameter_billions || 'unknown'}B</p>
        <button disabled={!!downloading} onClick={() => download(rec).catch((err) => setStatus(String(err)))}>Download</button>
      </article>)}
    </section>
  </main>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
