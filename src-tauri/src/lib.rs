use futures_util::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    env,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};
use sysinfo::{Disks, System};
use tauri::{Emitter, Manager, State};
use tokio::{fs::File, io::AsyncWriteExt};

const LOCAL_HOST: &str = "127.0.0.1";
const LOCAL_PORT: u16 = 39281;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os_name: String,
    pub arch: String,
    pub cpu_cores: usize,
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub disk_free_gb: f64,
    pub gpu_name: Option<String>,
    pub vram_gb: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendationRequest {
    pub task: String,
    pub query: String,
    pub speed_quality_preference: f64,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub repo_id: String,
    pub filename: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub repo_id: String,
    pub filename: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadResult {
    pub path: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub repo_id: String,
    pub filename: String,
    pub quantization: Option<String>,
    pub parameter_billions: Option<f64>,
    pub size_gb: Option<f64>,
    pub score: f64,
    pub fit_score: f64,
    pub quality_score: f64,
    pub speed_score: f64,
    pub safety_margin_score: f64,
    pub estimated_memory_gb: f64,
    pub gpu_layers: i32,
    pub context_size: u32,
    pub decision: String,
    pub reasons: Vec<String>,
    pub cautions: Vec<String>,
    pub download_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeLaunchRequest {
    pub model_path: String,
    pub gpu_layers: Option<i32>,
    pub context_size: Option<u32>,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub state: String,
    pub model_path: Option<String>,
    pub endpoint: String,
    pub detail: String,
    pub pid: Option<u32>,
    pub gpu_layers: Option<i32>,
    pub context_size: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

#[derive(Debug)]
struct RuntimeProcess {
    child: Child,
    model_path: String,
    gpu_layers: i32,
    context_size: u32,
}

#[derive(Default)]
struct RuntimeManager {
    process: Mutex<Option<RuntimeProcess>>,
}

#[derive(Debug, Clone)]
struct Candidate {
    repo_id: String,
    filename: String,
    quantization: Option<String>,
    parameter_billions: Option<f64>,
    size_gb: Option<f64>,
    downloads: u64,
    likes: u64,
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfModel {
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    id: Option<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    tags: Option<Vec<String>>,
    siblings: Option<Vec<HfSibling>>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: Option<String>,
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct LlamaChatResponse {
    choices: Vec<LlamaChoice>,
    usage: Option<LlamaUsage>,
}

#[derive(Debug, Deserialize)]
struct LlamaChoice {
    message: LlamaMessage,
}

#[derive(Debug, Deserialize)]
struct LlamaMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct LlamaUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

#[tauri::command]
fn scan_hardware() -> HardwareProfile {
    detect_hardware()
}

#[tauri::command]
async fn recommend_models(request: RecommendationRequest) -> Result<Vec<Recommendation>, String> {
    let hardware = detect_hardware();
    let candidates = search_hugging_face(&request).await?;
    let mut scored: Vec<Recommendation> = candidates
        .iter()
        .map(|candidate| score_candidate(candidate, &hardware, &request))
        .collect();
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(request.limit.max(1).min(30));
    Ok(scored)
}

#[tauri::command]
async fn download_model(app: tauri::AppHandle, request: DownloadRequest) -> Result<DownloadResult, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|err| format!("Could not resolve application data folder: {err}"))?
        .join("models")
        .join(sanitize(&request.repo_id));
    tokio::fs::create_dir_all(&base)
        .await
        .map_err(|err| format!("Could not create model folder: {err}"))?;

    let target = base.join(&request.filename);
    let partial = target.with_extension("partial");
    let client = reqwest::Client::new();
    let mut builder = client.get(&request.url).header("User-Agent", "MyAILocalModel/0.2");
    if let Ok(token) = env::var("HF_TOKEN").or_else(|_| env::var("HUGGINGFACE_TOKEN")) {
        builder = builder.bearer_auth(token);
    }
    let response = builder
        .send()
        .await
        .map_err(|err| format!("Could not contact Hugging Face: {err}"))?
        .error_for_status()
        .map_err(|err| format!("Hugging Face rejected the download: {err}"))?;

    let total = response.content_length();
    let mut file = File::create(&partial)
        .await
        .map_err(|err| format!("Could not create the model file: {err}"))?;
    let mut stream = response.bytes_stream();
    let mut downloaded = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| format!("Download interrupted: {err}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|err| format!("Could not write the downloaded model: {err}"))?;
        downloaded += chunk.len() as u64;
        let _ = app.emit(
            "download-progress",
            DownloadProgress {
                repo_id: request.repo_id.clone(),
                filename: request.filename.clone(),
                downloaded,
                total,
                percent: total.map(|value| downloaded as f64 / value as f64),
            },
        );
    }
    file.flush().await.map_err(|err| format!("Could not finish the model file: {err}"))?;
    tokio::fs::rename(&partial, &target)
        .await
        .map_err(|err| format!("Could not finalize the model file: {err}"))?;
    Ok(DownloadResult { path: target.to_string_lossy().to_string(), bytes: downloaded })
}

#[tauri::command]
async fn start_runtime(
    app: tauri::AppHandle,
    manager: State<'_, RuntimeManager>,
    request: RuntimeLaunchRequest,
) -> Result<RuntimeStatus, String> {
    stop_runtime_internal(&manager)?;

    let model = PathBuf::from(&request.model_path);
    if !model.exists() {
        return Err(format!("The selected model file does not exist: {}", model.display()));
    }

    let hardware = detect_hardware();
    let gpu_layers = request.gpu_layers.unwrap_or_else(|| recommended_gpu_layers(&hardware));
    let context_size = request.context_size.unwrap_or_else(|| recommended_context_size(&hardware));
    let threads = request.threads.unwrap_or_else(|| hardware.cpu_cores.saturating_sub(1).max(1));
    let executable = locate_llama_server(&app)?;

    let logs = app
        .path()
        .app_log_dir()
        .map_err(|err| format!("Could not resolve the log folder: {err}"))?;
    std::fs::create_dir_all(&logs).map_err(|err| format!("Could not create the log folder: {err}"))?;
    let stdout_file = std::fs::File::create(logs.join("llama-server.log"))
        .map_err(|err| format!("Could not create the runtime log: {err}"))?;
    let stderr_file = stdout_file
        .try_clone()
        .map_err(|err| format!("Could not prepare the runtime log: {err}"))?;

    let mut command = Command::new(&executable);
    command
        .arg("--model").arg(&model)
        .arg("--host").arg(LOCAL_HOST)
        .arg("--port").arg(LOCAL_PORT.to_string())
        .arg("--ctx-size").arg(context_size.to_string())
        .arg("--threads").arg(threads.to_string())
        .arg("--n-gpu-layers").arg(gpu_layers.to_string())
        .arg("--jinja")
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file));

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }

    let child = command.spawn().map_err(|err| {
        format!(
            "The bundled llama.cpp runtime could not start. Runtime: {}. Error: {err}",
            executable.display()
        )
    })?;
    let pid = child.id();
    {
        let mut guard = manager.process.lock().map_err(|_| "Runtime lock was poisoned".to_string())?;
        *guard = Some(RuntimeProcess {
            child,
            model_path: request.model_path.clone(),
            gpu_layers,
            context_size,
        });
    }

    let endpoint = local_endpoint();
    wait_for_health(&endpoint, Duration::from_secs(90)).await.map_err(|err| {
        let _ = stop_runtime_internal(&manager);
        format!("The model runtime started but did not become ready: {err}. Check the llama-server log in the application log folder.")
    })?;

    Ok(RuntimeStatus {
        state: "ready".to_string(),
        model_path: Some(request.model_path),
        endpoint,
        detail: "The local model is loaded and ready. No prompt data leaves this computer.".to_string(),
        pid: Some(pid),
        gpu_layers: Some(gpu_layers),
        context_size: Some(context_size),
    })
}

#[tauri::command]
fn stop_runtime(manager: State<'_, RuntimeManager>) -> Result<RuntimeStatus, String> {
    stop_runtime_internal(&manager)?;
    Ok(stopped_status("The local model runtime was stopped safely."))
}

#[tauri::command]
async fn restart_runtime(
    app: tauri::AppHandle,
    manager: State<'_, RuntimeManager>,
) -> Result<RuntimeStatus, String> {
    let request = {
        let guard = manager.process.lock().map_err(|_| "Runtime lock was poisoned".to_string())?;
        let process = guard.as_ref().ok_or_else(|| "No model has been started yet.".to_string())?;
        RuntimeLaunchRequest {
            model_path: process.model_path.clone(),
            gpu_layers: Some(process.gpu_layers),
            context_size: Some(process.context_size),
            threads: None,
        }
    };
    start_runtime(app, manager, request).await
}

#[tauri::command]
async fn runtime_status(manager: State<'_, RuntimeManager>) -> Result<RuntimeStatus, String> {
    let snapshot = {
        let mut guard = manager.process.lock().map_err(|_| "Runtime lock was poisoned".to_string())?;
        match guard.as_mut() {
            None => None,
            Some(process) => match process.child.try_wait() {
                Ok(Some(exit)) => {
                    let detail = format!("The local runtime exited unexpectedly with {exit}.");
                    *guard = None;
                    return Ok(stopped_status(&detail));
                }
                Ok(None) => Some((process.child.id(), process.model_path.clone(), process.gpu_layers, process.context_size)),
                Err(err) => return Err(format!("Could not inspect the local runtime: {err}")),
            },
        }
    };

    if let Some((pid, model_path, gpu_layers, context_size)) = snapshot {
        let endpoint = local_endpoint();
        let ready = reqwest::Client::new()
            .get(format!("{endpoint}/health"))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false);
        Ok(RuntimeStatus {
            state: if ready { "ready" } else { "loading" }.to_string(),
            model_path: Some(model_path),
            endpoint,
            detail: if ready {
                "The local model is ready.".to_string()
            } else {
                "The runtime is running and the model is still loading.".to_string()
            },
            pid: Some(pid),
            gpu_layers: Some(gpu_layers),
            context_size: Some(context_size),
        })
    } else {
        Ok(stopped_status("No local model is running."))
    }
}

#[tauri::command]
async fn chat(request: ChatRequest, manager: State<'_, RuntimeManager>) -> Result<ChatResponse, String> {
    {
        let mut guard = manager.process.lock().map_err(|_| "Runtime lock was poisoned".to_string())?;
        let process = guard.as_mut().ok_or_else(|| "Start a local model before sending a message.".to_string())?;
        if let Some(exit) = process.child.try_wait().map_err(|err| format!("Could not inspect runtime: {err}"))? {
            *guard = None;
            return Err(format!("The model runtime stopped before the request could be sent: {exit}"));
        }
    }

    let payload = serde_json::json!({
        "messages": request.messages,
        "temperature": request.temperature.unwrap_or(0.7),
        "max_tokens": request.max_tokens.unwrap_or(768),
        "stream": false
    });
    let response = reqwest::Client::new()
        .post(format!("{}/v1/chat/completions", local_endpoint()))
        .json(&payload)
        .timeout(Duration::from_secs(300))
        .send()
        .await
        .map_err(|err| format!("The local model could not answer: {err}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("The local model returned {status}: {body}"));
    }
    let parsed: LlamaChatResponse = response
        .json()
        .await
        .map_err(|err| format!("The local response could not be read: {err}"))?;
    let content = parsed
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .ok_or_else(|| "The local model returned no answer.".to_string())?;
    Ok(ChatResponse {
        content,
        prompt_tokens: parsed.usage.as_ref().and_then(|usage| usage.prompt_tokens),
        completion_tokens: parsed.usage.as_ref().and_then(|usage| usage.completion_tokens),
    })
}

fn locate_llama_server(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let exe_name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    let mut candidates = Vec::new();
    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("runtime").join(exe_name));
        candidates.push(resource_dir.join(exe_name));
    }
    if let Ok(app_data) = app.path().app_data_dir() {
        candidates.push(app_data.join("runtime").join(exe_name));
    }
    if let Ok(current) = env::current_dir() {
        candidates.push(current.join("src-tauri").join("resources").join("runtime").join(exe_name));
        candidates.push(current.join("resources").join("runtime").join(exe_name));
    }
    candidates
        .into_iter()
        .find(|path| path.exists())
        .ok_or_else(|| "The llama.cpp runtime is missing from this installation. Reinstall MyAILocalModel or use an official release installer.".to_string())
}

fn stop_runtime_internal(manager: &RuntimeManager) -> Result<(), String> {
    let mut guard = manager.process.lock().map_err(|_| "Runtime lock was poisoned".to_string())?;
    if let Some(mut process) = guard.take() {
        let _ = process.child.kill();
        let _ = process.child.wait();
    }
    Ok(())
}

async fn wait_for_health(endpoint: &str, timeout: Duration) -> Result<(), String> {
    let started = Instant::now();
    let client = reqwest::Client::new();
    while started.elapsed() < timeout {
        match client
            .get(format!("{endpoint}/health"))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            _ => tokio::time::sleep(Duration::from_millis(750)).await,
        }
    }
    Err(format!("Timed out after {} seconds", timeout.as_secs()))
}

fn local_endpoint() -> String {
    format!("http://{LOCAL_HOST}:{LOCAL_PORT}")
}

fn stopped_status(detail: &str) -> RuntimeStatus {
    RuntimeStatus {
        state: "stopped".to_string(),
        model_path: None,
        endpoint: local_endpoint(),
        detail: detail.to_string(),
        pid: None,
        gpu_layers: None,
        context_size: None,
    }
}

fn detect_hardware() -> HardwareProfile {
    let mut system = System::new_all();
    system.refresh_all();
    let total_ram_gb = bytes_to_gb(system.total_memory());
    let available_ram_gb = bytes_to_gb(system.available_memory());
    let cpu_cores = system.cpus().len().max(1);
    let disks = Disks::new_with_refreshed_list();
    let disk_free_gb = disks.iter().map(|disk| disk.available_space()).max().map(bytes_to_gb).unwrap_or(0.0);
    let mut notes = Vec::new();
    let (gpu_name, vram_gb) = detect_gpu(&mut notes);
    if gpu_name.is_none() {
        notes.push("No discrete GPU was detected. The setup wizard will use conservative CPU-friendly settings.".to_string());
    }
    HardwareProfile {
        os_name: System::name().unwrap_or_else(|| env::consts::OS.to_string()),
        arch: env::consts::ARCH.to_string(),
        cpu_cores,
        total_ram_gb,
        available_ram_gb,
        disk_free_gb,
        gpu_name,
        vram_gb,
        notes,
    }
}

#[cfg(windows)]
fn detect_gpu(notes: &mut Vec<String>) -> (Option<String>, Option<f64>) {
    use serde::Deserialize;
    use wmi::{COMLibrary, WMIConnection};

    #[derive(Debug, Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct VideoController {
        name: Option<String>,
        adapter_ram: Option<u64>,
    }

    let com = match COMLibrary::new() { Ok(value) => value, Err(_) => return (None, None) };
    let wmi = match WMIConnection::new(com.into()) { Ok(value) => value, Err(_) => return (None, None) };
    let devices: Vec<VideoController> = match wmi.query() { Ok(value) => value, Err(_) => return (None, None) };
    let mut best_name = None;
    let mut best_vram = None;
    for device in devices {
        let Some(name) = device.name else { continue; };
        if name.to_lowercase().contains("basic display") { continue; }
        let vram = device.adapter_ram.map(bytes_to_gb);
        if best_vram.unwrap_or(0.0) <= vram.unwrap_or(0.0) {
            best_name = Some(name);
            best_vram = vram;
        }
    }
    if best_name.is_some() && best_vram.is_none() {
        notes.push("Windows detected a GPU but did not report VRAM. The wizard will choose RAM-first settings.".to_string());
    }
    (best_name, best_vram)
}

#[cfg(not(windows))]
fn detect_gpu(_notes: &mut Vec<String>) -> (Option<String>, Option<f64>) { (None, None) }

async fn search_hugging_face(request: &RecommendationRequest) -> Result<Vec<Candidate>, String> {
    let task_hint = match request.task.as_str() {
        "coding" => "GGUF coder instruct",
        "writing" => "GGUF instruct llama mistral",
        "research" => "GGUF reasoning instruct qwen deepseek",
        "small-business" => "GGUF instruct small",
        _ => request.query.as_str(),
    };
    let url = reqwest::Url::parse_with_params(
        "https://huggingface.co/api/models",
        &[("search", task_hint), ("sort", "downloads"), ("direction", "-1"), ("limit", "50"), ("full", "true")],
    ).map_err(|err| format!("Could not build the Hugging Face search: {err}"))?;
    let mut builder = reqwest::Client::new().get(url).header("User-Agent", "MyAILocalModel/0.2");
    if let Ok(token) = env::var("HF_TOKEN").or_else(|_| env::var("HUGGINGFACE_TOKEN")) {
        builder = builder.bearer_auth(token);
    }
    let models: Vec<HfModel> = builder.send().await
        .map_err(|err| format!("Hugging Face search failed: {err}"))?
        .error_for_status().map_err(|err| format!("Hugging Face rejected the search: {err}"))?
        .json().await.map_err(|err| format!("Could not read the Hugging Face response: {err}"))?;

    let mut candidates = Vec::new();
    for model in models {
        let Some(repo_id) = model.model_id.or(model.id) else { continue; };
        let tags = model.tags.unwrap_or_default();
        for sibling in model.siblings.unwrap_or_default() {
            let Some(filename) = sibling.rfilename else { continue; };
            let lower = filename.to_lowercase();
            if !lower.ends_with(".gguf") || lower.contains("mmproj") || lower.contains("tokenizer") { continue; }
            candidates.push(Candidate {
                repo_id: repo_id.clone(),
                filename: filename.clone(),
                quantization: extract_quantization(&filename),
                parameter_billions: extract_params(&format!("{repo_id}/{filename}")),
                size_gb: sibling.size.map(bytes_to_gb),
                downloads: model.downloads.unwrap_or(0),
                likes: model.likes.unwrap_or(0),
                tags: tags.clone(),
            });
        }
    }
    if candidates.is_empty() {
        return Err("No compatible GGUF model files were found. Try a broader search such as 'GGUF instruct'.".to_string());
    }
    Ok(candidates)
}

fn score_candidate(candidate: &Candidate, hardware: &HardwareProfile, request: &RecommendationRequest) -> Recommendation {
    let estimated_memory_gb = estimate_memory(candidate);
    let effective_ram = hardware.available_ram_gb.min(hardware.total_ram_gb * 0.72).max(1.0);
    let fit_score: f64 = if estimated_memory_gb <= effective_ram * 0.75 { 1.0 } else if estimated_memory_gb <= effective_ram { 0.72 } else if estimated_memory_gb <= hardware.total_ram_gb * 0.85 { 0.38 } else { 0.08 };
    let quality_score = quality_score(candidate);
    let quant_speed: f64 = match candidate.quantization.as_deref().unwrap_or("Q4") {
        q if q.starts_with("Q2") => 0.95,
        q if q.starts_with("Q3") => 0.90,
        q if q.starts_with("Q4") => 0.84,
        q if q.starts_with("Q5") => 0.72,
        q if q.starts_with("Q6") => 0.62,
        q if q.starts_with("Q8") => 0.46,
        "F16" => 0.25,
        _ => 0.70,
    };
    let vram = hardware.vram_gb.unwrap_or(0.0);
    let gpu_offload = vram >= estimated_memory_gb * 0.72;
    let size_penalty = (candidate.parameter_billions.unwrap_or(7.0) / 20.0).min(1.0) * 0.35;
    let speed_score = (quant_speed - size_penalty + if gpu_offload { 0.20 } else { 0.0 }).clamp(0.05, 1.0);
    let safety_margin_score = (((effective_ram - estimated_memory_gb) / effective_ram) * 1.5).clamp(0.0, 1.0);
    let popularity = ((candidate.downloads.max(1) as f64).log10() / 6.0 + candidate.likes.min(500) as f64 / 3000.0).min(1.0);
    let preference = request.speed_quality_preference.clamp(0.0, 1.0);
    let task_bonus = if task_matches(candidate, &request.task) { 0.06 } else { 0.0 };
    let score = (fit_score * 0.37 + quality_score * (0.12 + preference * 0.18) + speed_score * (0.25 - preference * 0.12) + safety_margin_score * 0.15 + popularity * 0.08 + task_bonus).clamp(0.0, 1.0);
    let gpu_layers = if gpu_offload { 999 } else if vram >= 4.0 { 20 } else { 0 };
    let context_size = recommended_context_size(hardware);
    let mut reasons = vec![format!("Estimated loaded footprint is about {:.1} GB including runtime overhead.", estimated_memory_gb)];
    let mut cautions = Vec::new();
    if fit_score >= 0.7 { reasons.push("The model fits within the memory currently available on this computer.".to_string()); }
    else { cautions.push("Memory headroom is tight. Close other applications or choose a smaller quantization.".to_string()); }
    if gpu_offload { reasons.push(format!("The detected {:.1} GB of VRAM should support substantial GPU acceleration.", vram)); }
    else if hardware.gpu_name.is_some() { cautions.push("Only partial GPU offload is likely; generation may lean heavily on the CPU.".to_string()); }
    else { cautions.push("No compatible GPU was detected, so generation will run primarily on the CPU.".to_string()); }
    if let Some(q) = &candidate.quantization { reasons.push(format!("{q} is an appropriate speed, memory, and quality tradeoff for this hardware.")); }
    let decision = if fit_score < 0.4 { "Not recommended for this computer." } else if speed_score > quality_score { "Best for fast, low-friction local chat." } else if quality_score > 0.82 { "Best quality option that still appears to fit." } else { "Balanced recommendation for this computer." };
    let encoded = candidate.filename.replace(' ', "%20").replace('#', "%23");
    Recommendation {
        repo_id: candidate.repo_id.clone(),
        filename: candidate.filename.clone(),
        quantization: candidate.quantization.clone(),
        parameter_billions: candidate.parameter_billions,
        size_gb: candidate.size_gb,
        score: round3(score),
        fit_score: round3(fit_score),
        quality_score: round3(quality_score),
        speed_score: round3(speed_score),
        safety_margin_score: round3(safety_margin_score),
        estimated_memory_gb: round3(estimated_memory_gb),
        gpu_layers,
        context_size,
        decision: decision.to_string(),
        reasons,
        cautions,
        download_url: format!("https://huggingface.co/{}/resolve/main/{}", candidate.repo_id, encoded),
    }
}

fn estimate_memory(candidate: &Candidate) -> f64 {
    if let Some(size) = candidate.size_gb { return size * 1.25 + 0.75; }
    let params = candidate.parameter_billions.unwrap_or(7.0);
    let multiplier = match candidate.quantization.as_deref().unwrap_or("Q4") {
        q if q.starts_with("Q2") => 0.32,
        q if q.starts_with("Q3") => 0.42,
        q if q.starts_with("Q4") => 0.55,
        q if q.starts_with("Q5") => 0.68,
        q if q.starts_with("Q6") => 0.80,
        q if q.starts_with("Q8") => 1.05,
        "F16" => 2.05,
        _ => 0.55,
    };
    params * multiplier + 1.25
}

fn quality_score(candidate: &Candidate) -> f64 {
    let text = format!("{} {} {}", candidate.repo_id, candidate.filename, candidate.tags.join(" ")).to_lowercase();
    let family: f64 = if text.contains("deepseek") { 0.91 } else if text.contains("qwen") { 0.89 } else if text.contains("llama") { 0.88 } else if text.contains("mistral") { 0.87 } else if text.contains("gemma") { 0.82 } else if text.contains("phi") { 0.78 } else { 0.72 };
    (family * 0.82 + (candidate.parameter_billions.unwrap_or(3.0) / 14.0).min(1.0) * 0.18).min(1.0)
}

fn task_matches(candidate: &Candidate, task: &str) -> bool {
    let text = format!("{} {} {}", candidate.repo_id, candidate.filename, candidate.tags.join(" ")).to_lowercase();
    match task {
        "coding" => text.contains("coder") || text.contains("code"),
        "writing" => text.contains("llama") || text.contains("mistral") || text.contains("story"),
        "research" => text.contains("deepseek") || text.contains("qwen") || text.contains("reason"),
        "small-business" => text.contains("instruct") || text.contains("chat"),
        _ => true,
    }
}

fn recommended_gpu_layers(hardware: &HardwareProfile) -> i32 {
    match hardware.vram_gb.unwrap_or(0.0) {
        v if v >= 10.0 => 999,
        v if v >= 6.0 => 28,
        v if v >= 4.0 => 16,
        _ => 0,
    }
}

fn recommended_context_size(hardware: &HardwareProfile) -> u32 {
    if hardware.total_ram_gb >= 64.0 { 16384 } else if hardware.total_ram_gb >= 32.0 { 8192 } else { 4096 }
}

fn extract_quantization(filename: &str) -> Option<String> {
    Regex::new(r"(?i)(Q[2-8](?:_[A-Z0-9]+)?|F16)").ok()?.captures(filename).and_then(|c| c.get(1)).map(|m| m.as_str().to_uppercase())
}

fn extract_params(text: &str) -> Option<f64> {
    Regex::new(r"(?i)(?:^|[-_/.])([0-9]+(?:\.[0-9]+)?)b(?:[-_/.]|$)").ok()?.captures(text).and_then(|c| c.get(1)).and_then(|m| m.as_str().parse().ok())
}

fn bytes_to_gb(value: u64) -> f64 { (value as f64 / 1024_f64.powi(3) * 10.0).round() / 10.0 }
fn round3(value: f64) -> f64 { (value * 1000.0).round() / 1000.0 }
fn sanitize(value: &str) -> String { value.replace(['/', '\\', ':'], "_") }

pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeManager::default())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            scan_hardware,
            recommend_models,
            download_model,
            start_runtime,
            stop_runtime,
            restart_runtime,
            runtime_status,
            chat
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let manager = window.state::<RuntimeManager>();
                let _ = stop_runtime_internal(&manager);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running MyAILocalModel");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hardware() -> HardwareProfile {
        HardwareProfile { os_name: "Windows".into(), arch: "x86_64".into(), cpu_cores: 12, total_ram_gb: 32.0, available_ram_gb: 20.0, disk_free_gb: 500.0, gpu_name: Some("NVIDIA RTX 4070".into()), vram_gb: Some(12.0), notes: vec![] }
    }

    fn candidate(name: &str, file: &str, params: f64, size: f64) -> Candidate {
        Candidate { repo_id: name.into(), filename: file.into(), quantization: extract_quantization(file), parameter_billions: Some(params), size_gb: Some(size), downloads: 100000, likes: 200, tags: vec!["gguf".into(), "instruct".into()] }
    }

    #[test]
    fn practical_q4_model_scores_as_a_good_fit() {
        let req = RecommendationRequest { task: "general".into(), query: "GGUF instruct".into(), speed_quality_preference: 0.5, limit: 5 };
        let rec = score_candidate(&candidate("Qwen/Qwen-7B-GGUF", "qwen-7b.Q4_K_M.gguf", 7.0, 4.8), &hardware(), &req);
        assert!(rec.fit_score > 0.7);
        assert!(rec.gpu_layers > 0);
    }

    #[test]
    fn oversized_model_is_rejected() {
        let req = RecommendationRequest { task: "general".into(), query: "GGUF instruct".into(), speed_quality_preference: 0.9, limit: 5 };
        let rec = score_candidate(&candidate("Meta/Llama-70B-GGUF", "llama-70b.Q8_0.gguf", 70.0, 74.0), &hardware(), &req);
        assert!(rec.fit_score < 0.4);
        assert!(rec.decision.contains("Not recommended"));
    }

    #[test]
    fn context_defaults_scale_with_ram() {
        assert_eq!(recommended_context_size(&hardware()), 8192);
    }
}
