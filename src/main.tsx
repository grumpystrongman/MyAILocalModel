import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import './styles.css';

type Hardware = { os_name:string; arch:string; cpu_cores:number; total_ram_gb:number; available_ram_gb:number; disk_free_gb:number; gpu_name?:string|null; vram_gb?:number|null; notes:string[] };
type Rec = { repo_id:string; filename:string; quantization?:string|null; parameter_billions?:number|null; size_gb?:number|null; score:number; fit_score:number; quality_score:number; speed_score:number; safety_margin_score:number; estimated_memory_gb:number; gpu_layers:number; context_size:number; decision:string; reasons:string[]; cautions:string[]; download_url:string };
type RuntimeStatus = { state:string; model_path?:string|null; endpoint:string; detail:string; pid?:number|null; gpu_layers?:number|null; context_size?:number|null };
type ChatMessage = { role:'system'|'user'|'assistant'; content:string };
type DownloadProgress = { downloaded:number; total?:number|null; percent?:number|null };

const tasks = [
  ['general','General chat'],['coding','Coding'],['writing','Writing'],['research','Research'],['small-business','Private business AI']
];

function App(){
  const [step,setStep]=useState(1);
  const [hardware,setHardware]=useState<Hardware|null>(null);
  const [task,setTask]=useState('general');
  const [preference,setPreference]=useState(.5);
  const [recs,setRecs]=useState<Rec[]>([]);
  const [selected,setSelected]=useState<Rec|null>(null);
  const [modelPath,setModelPath]=useState<string|null>(null);
  const [runtime,setRuntime]=useState<RuntimeStatus|null>(null);
  const [status,setStatus]=useState('Preparing setup...');
  const [progress,setProgress]=useState<number|null>(null);
  const [messages,setMessages]=useState<ChatMessage[]>([{role:'system',content:'You are a helpful private assistant running entirely on this computer.'}]);
  const [input,setInput]=useState('');
  const [busy,setBusy]=useState(false);

  useEffect(()=>{
    const boot=async()=>{
      const hw=await invoke<Hardware>('scan_hardware'); setHardware(hw); setStatus('Hardware scan complete.');
      const rs=await invoke<RuntimeStatus>('runtime_status'); setRuntime(rs);
    };
    boot().catch(e=>setStatus(String(e)));
    const unlisten=listen<DownloadProgress>('download-progress',e=>setProgress(e.payload.percent??null));
    return ()=>{unlisten.then(fn=>fn());};
  },[]);

  const top=selected??recs[0]??null;
  const pct=(n:number)=>`${Math.round(n*100)}%`;

  async function findModels(){
    setBusy(true); setStatus('Searching Hugging Face and comparing models...');
    try{
      const result=await invoke<Rec[]>('recommend_models',{request:{task,query:'GGUF instruct',speed_quality_preference:preference,limit:10}});
      setRecs(result); setSelected(result[0]??null); setStep(3); setStatus('Recommendation ready.');
    }finally{setBusy(false);}
  }

  async function installAndStart(){
    if(!top)return;
    setBusy(true); setProgress(0); setStatus(`Downloading ${top.filename}...`);
    try{
      const result=await invoke<{path:string;bytes:number}>('download_model',{request:{repo_id:top.repo_id,filename:top.filename,url:top.download_url}});
      setModelPath(result.path); setStatus('Model downloaded. Starting the private runtime...');
      const rs=await invoke<RuntimeStatus>('start_runtime',{request:{model_path:result.path,gpu_layers:top.gpu_layers,context_size:top.context_size,threads:null}});
      setRuntime(rs); setStep(4); setStatus(rs.detail);
    }finally{setBusy(false); setProgress(null);}
  }

  async function send(){
    const text=input.trim(); if(!text||busy)return;
    const next=[...messages,{role:'user' as const,content:text}]; setMessages(next); setInput(''); setBusy(true); setStatus('Thinking locally...');
    try{
      const response=await invoke<{content:string}>('chat',{request:{messages:next,temperature:.7,max_tokens:768}});
      setMessages([...next,{role:'assistant',content:response.content}]); setStatus('Ready.');
    }catch(e){setStatus(String(e));}
    finally{setBusy(false);}
  }

  async function stop(){const rs=await invoke<RuntimeStatus>('stop_runtime');setRuntime(rs);setStatus(rs.detail);}
  async function restart(){setBusy(true);try{const rs=await invoke<RuntimeStatus>('restart_runtime');setRuntime(rs);setStatus(rs.detail);}finally{setBusy(false);}}

  return <main>
    <header className="topbar"><div><span className="brand">MyAILocalModel</span><span className="privacy">Private by default</span></div><span className={`runtime ${runtime?.state??'stopped'}`}>{runtime?.state??'stopped'}</span></header>

    {step<4 && <section className="wizard">
      <div className="steps"><span className={step>=1?'active':''}>1 Scan</span><span className={step>=2?'active':''}>2 Purpose</span><span className={step>=3?'active':''}>3 Choose</span><span>4 Chat</span></div>

      {step===1&&<div className="panel hero"><p className="eyebrow">Welcome</p><h1>Let’s find the best private AI for this computer.</h1><p>No terminal. No configuration files. Your prompts stay local.</p>{hardware&&<div className="spec-grid"><b>Memory</b><span>{hardware.available_ram_gb.toFixed(1)} GB available of {hardware.total_ram_gb.toFixed(1)} GB</span><b>Processor</b><span>{hardware.cpu_cores} cores</span><b>Graphics</b><span>{hardware.gpu_name||'CPU mode'} {hardware.vram_gb?`· ${hardware.vram_gb.toFixed(1)} GB VRAM`:''}</span><b>Storage</b><span>{hardware.disk_free_gb.toFixed(1)} GB free</span></div>}<button className="primary" onClick={()=>setStep(2)} disabled={!hardware}>Continue</button></div>}

      {step===2&&<div className="panel"><p className="eyebrow">What will you use it for?</p><h2>Choose the work that matters most.</h2><div className="choice-grid">{tasks.map(([id,label])=><button key={id} className={task===id?'choice selected':'choice'} onClick={()=>setTask(id)}>{label}</button>)}</div><label className="slider">Faster responses <input type="range" min="0" max="1" step=".05" value={preference} onChange={e=>setPreference(Number(e.target.value))}/> Better answers</label><button className="primary" disabled={busy} onClick={()=>findModels().catch(e=>setStatus(String(e)))}>{busy?'Comparing models...':'Find my best model'}</button></div>}

      {step===3&&top&&<div className="panel"><p className="eyebrow">Best match for this computer</p><h2>{top.repo_id}</h2><p className="filename">{top.filename}</p><p className="decision">{top.decision}</p><div className="score-grid"><Score label="Fit" value={top.fit_score}/><Score label="Quality" value={top.quality_score}/><Score label="Speed" value={top.speed_score}/><Score label="Headroom" value={top.safety_margin_score}/></div><div className="why"><div><h3>Why it was chosen</h3>{top.reasons.map(r=><p key={r}>✓ {r}</p>)}</div><div><h3>What to know</h3>{top.cautions.length?top.cautions.map(c=><p key={c}>⚠ {c}</p>):<p>No major cautions.</p>}</div></div><p><b>Automatic settings:</b> {top.gpu_layers===999?'full GPU offload':`${top.gpu_layers} GPU layers`}, {top.context_size.toLocaleString()} token context.</p><div className="model-list">{recs.slice(0,5).map(r=><button key={r.filename} onClick={()=>setSelected(r)} className={top.filename===r.filename?'model-option selected':'model-option'}><span>{r.repo_id}</span><small>{r.quantization||'GGUF'} · {r.estimated_memory_gb.toFixed(1)} GB estimated</small></button>)}</div>{progress!==null&&<div className="download"><progress max="1" value={progress}/><span>{Math.round(progress*100)}%</span></div>}<button className="primary" disabled={busy} onClick={()=>installAndStart().catch(e=>setStatus(String(e)))}>{busy?'Installing and loading...':'Install model and start private AI'}</button></div>}
    </section>}

    {step===4&&<section className="chat-shell"><aside><h2>Local AI</h2><p>{top?.repo_id}</p><div className="runtime-card"><b>{runtime?.state==='ready'?'Ready':'Not ready'}</b><span>{runtime?.detail}</span><small>Context: {runtime?.context_size?.toLocaleString()??'—'} · GPU layers: {runtime?.gpu_layers??'—'}</small></div><button onClick={()=>restart().catch(e=>setStatus(String(e)))} disabled={busy}>Restart model</button><button onClick={()=>stop().catch(e=>setStatus(String(e)))}>Stop model</button><button onClick={()=>setStep(2)}>Choose another model</button></aside><section className="chat"><div className="messages">{messages.filter(m=>m.role!=='system').map((m,i)=><div key={i} className={`message ${m.role}`}><b>{m.role==='user'?'You':'Local AI'}</b><p>{m.content}</p></div>)}{messages.length===1&&<div className="empty"><h2>Your private AI is ready.</h2><p>Ask a question. The answer is generated on this computer.</p></div>}</div><div className="composer"><textarea value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}} placeholder="Message your local model..."/><button className="primary" onClick={send} disabled={busy||runtime?.state!=='ready'}>{busy?'Working...':'Send'}</button></div></section></section>}

    <footer className="statusbar"><span>{status}</span><span>Local endpoint: {runtime?.endpoint??'not started'}</span></footer>
  </main>;
}

function Score({label,value}:{label:string;value:number}){return <div className="score"><span>{label}</span><b>{Math.round(value*100)}%</b><meter min="0" max="1" value={value}/></div>}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
