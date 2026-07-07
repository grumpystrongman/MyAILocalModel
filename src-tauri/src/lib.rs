use futures_util::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{env, fs, path::PathBuf};
use sysinfo::{Disks, System};
use tauri::{Emitter, Manager};
use tokio::{fs::File, io::AsyncWriteExt};

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
    pub popularity_score: f64,
    pub usability_score: f64,
    pub estimated_memory_gb: f64,
    pub expected_runtime: String,
    pub runtime_recommendations: Vec<String>,
    pub decision: String,
    pub reasons: Vec<String>,
    pub cautions: Vec<String>,
    pub download_url: String,
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
    updated_recently: bool,
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
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: Option<String>,
    size: Option<u64>,
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
        .map_err(|err| format!("Could not resolve app data directory: {err}"))?
        .join("models")
        .join(sanitize(&request.repo_id));
    tokio::fs::create_dir_all(&base)
        .await
        .map_err(|err| format!("Could not create model cache: {err}"))?;

    let target = base.join(&request.filename);
    let tmp = target.with_extension("partial");
    let client = reqwest::Client::new();
    let mut builder = client.get(&request.url);
    if let Ok(token) = env::var("HF_TOKEN").or_else(|_| env::var("HUGGINGFACE_TOKEN")) {
        builder = builder.bearer_auth(token);
    }
    let response = builder
        .send()
        .await
        .map_err(|err| format!("Hugging Face download request failed: {err}"))?
        .error_for_status()
        .map_err(|err| format!("Hugging Face rejected the download: {err}"))?;

    let total = response.content_length();
    let mut file = File::create(&tmp)
        .await
        .map_err(|err| format!("Could not write model file: {err}"))?;
    let mut stream = response.bytes_stream();
    let mut downloaded = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| format!("Download stream failed: {err}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|err| format!("Could not write chunk: {err}"))?;
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
    file.flush()
        .await
        .map_err(|err| format!("Could not flush model file: {err}"))?;
    tokio::fs::rename(&tmp, &target)
        .await
        .map_err(|err| format!("Could not finalize model file: {err}"))?;
    Ok(DownloadResult { path: target.to_string_lossy().to_string(), bytes: downloaded })
}

fn detect_hardware() -> HardwareProfile {
    let mut system = System::new_all();
    system.refresh_all();
    let total_ram_gb = bytes_to_gb(system.total_memory());
    let available_ram_gb = bytes_to_gb(system.available_memory());
    let cpu_cores = system.cpus().len().max(1);
    let disks = Disks::new_with_refreshed_list();
    let disk_free_gb = disks
        .iter()
        .map(|disk| disk.available_space())
        .max()
        .map(bytes_to_gb)
        .unwrap_or(0.0);
    let mut notes = Vec::new();
    let (gpu_name, vram_gb) = detect_gpu(&mut notes);
    if gpu_name.is_none() {
        notes.push("No discrete GPU was detected. Recommendations will favor CPU-friendly GGUF models and conservative quantization.".to_string());
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

    let com = match COMLibrary::new() {
        Ok(value) => value,
        Err(_) => return (None, None),
    };
    let wmi = match WMIConnection::new(com.into()) {
        Ok(value) => value,
        Err(_) => return (None, None),
    };
    let devices: Vec<VideoController> = match wmi.query() {
        Ok(value) => value,
        Err(_) => return (None, None),
    };
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
        notes.push("Windows detected a GPU, but VRAM was not reported by WMI. The advisor will use RAM-first scoring.".to_string());
    }
    (best_name, best_vram)
}

#[cfg(not(windows))]
fn detect_gpu(_notes: &mut Vec<String>) -> (Option<String>, Option<f64>) {
    (None, None)
}

async fn search_hugging_face(request: &RecommendationRequest) -> Result<Vec<Candidate>, String> {
    let task_hint = match request.task.as_str() {
        "coding" => "GGUF coder instruct",
        "writing" => "GGUF instruct llama mistral",
        "research" => "GGUF reasoning instruct qwen deepseek llama",
        "small-business" => "GGUF instruct small",
        _ => request.query.as_str(),
    };
    let url = reqwest::Url::parse_with_params(
        "https://huggingface.co/api/models",
        &[
            ("search", task_hint),
            ("sort", "downloads"),
            ("direction", "-1"),
            ("limit", "50"),
            ("full", "true"),
        ],
    )
    .map_err(|err| format!("Could not build Hugging Face query: {err}"))?;
    let client = reqwest::Client::new();
    let mut builder = client.get(url).header("User-Agent", "MyAILocalModel/0.1");
    if let Ok(token) = env::var("HF_TOKEN").or_else(|_| env::var("HUGGINGFACE_TOKEN")) {
        builder = builder.bearer_auth(token);
    }
    let models: Vec<HfModel> = builder
        .send()
        .await
        .map_err(|err| format!("Hugging Face search failed: {err}"))?
        .error_for_status()
        .map_err(|err| format!("Hugging Face search was rejected: {err}"))?
        .json()
        .await
        .map_err(|err| format!("Could not parse Hugging Face response: {err}"))?;

    let mut candidates = Vec::new();
    for model in models {
        let Some(repo_id) = model.model_id.or(model.id) else { continue; };
        let tags = model.tags.unwrap_or_default();
        let updated_recently = model.last_modified.as_deref().unwrap_or_default() >= "2024";
        for sibling in model.siblings.unwrap_or_default() {
            let Some(filename) = sibling.rfilename else { continue; };
            let lower = filename.to_lowercase();
            if !lower.ends_with(".gguf") || lower.contains("mmproj") || lower.contains("tokenizer") {
                continue;
            }
            candidates.push(Candidate {
                repo_id: repo_id.clone(),
                filename: filename.clone(),
                quantization: extract_quantization(&filename),
                parameter_billions: extract_params(&format!("{repo_id}/{filename}")),
                size_gb: sibling.size.map(bytes_to_gb),
                downloads: model.downloads.unwrap_or(0),
                likes: model.likes.unwrap_or(0),
                tags: tags.clone(),
                updated_recently,
            });
        }
    }
    if candidates.is_empty() {
        return Err("No GGUF artifacts found. Try a broader query like 'GGUF instruct'.".to_string());
    }
    Ok(candidates)
}

fn score_candidate(candidate: &Candidate, hardware: &HardwareProfile, request: &RecommendationRequest) -> Recommendation {
    let estimated_memory_gb = estimate_memory(candidate);
    let effective_ram = hardware.available_ram_gb.min(hardware.total_ram_gb * 0.72).max(1.0);
    let vram = hardware.vram_gb.unwrap_or(0.0);
    let gpu_offload_realistic = vram >= estimated_memory_gb * 0.72;

    let fit_score = if estimated_memory_gb <= effective_ram * 0.75 { 1.0 }
        else if estimated_memory_gb <= effective_ram { 0.72 }
        else if estimated_memory_gb <= hardware.total_ram_gb * 0.85 { 0.38 }
        else { 0.08 };

    let quality_score = quality_score(candidate);
    let quant_speed = match candidate.quantization.as_deref().unwrap_or("Q4") {
        q if q.starts_with("Q2") => 0.95,
        q if q.starts_with("Q3") => 0.90,
        q if q.starts_with("Q4") => 0.84,
        q if q.starts_with("Q5") => 0.72,
        q if q.starts_with("Q6") => 0.62,
        q if q.starts_with("Q8") => 0.46,
        "F16" => 0.25,
        _ => 0.70,
    };
    let size_penalty = (candidate.parameter_billions.unwrap_or(7.0) / 20.0).min(1.0) * 0.35;
    let speed_score = (quant_speed - size_penalty + if gpu_offload_realistic { 0.20 } else { 0.0 }).clamp(0.05, 1.0);
    let margin = (effective_ram - estimated_memory_gb) / effective_ram;
    let safety_margin_score = (margin * 1.5).clamp(0.0, 1.0);
    let popularity_score = ((candidate.downloads.max(1) as f64).log10() / 6.0 + (candidate.likes.min(500) as f64 / 3000.0)).min(1.0);
    let usability_score = usability_score(candidate);

    let quality_weight = 0.12 + request.speed_quality_preference.clamp(0.0, 1.0) * 0.18;
    let speed_weight = 0.25 - request.speed_quality_preference.clamp(0.0, 1.0) * 0.12;
    let mut score = fit_score * 0.34
        + quality_score * quality_weight
        + speed_score * speed_weight
        + safety_margin_score * 0.15
        + popularity_score * 0.08
        + usability_score * 0.06;
    if task_matches(candidate, &request.task) { score += 0.06; }
    if candidate.updated_recently { score += 0.02; }
    score = score.clamp(0.0, 1.0);

    let expected_runtime = choose_runtime(hardware, candidate, gpu_offload_realistic);
    let runtime_recommendations = runtime_recommendations(hardware, candidate);
    let (decision, reasons, cautions) = explain(candidate, hardware, estimated_memory_gb, fit_score, quality_score, speed_score, gpu_offload_realistic);
    let encoded_file = candidate.filename.replace('#', "%23").replace(' ', "%20");
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
        popularity_score: round3(popularity_score),
        usability_score: round3(usability_score),
        estimated_memory_gb: round3(estimated_memory_gb),
        expected_runtime,
        runtime_recommendations,
        decision,
        reasons,
        cautions,
        download_url: format!("https://huggingface.co/{}/resolve/main/{}", candidate.repo_id, encoded_file),
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
    let family = if text.contains("deepseek") { 0.91 }
        else if text.contains("qwen") { 0.89 }
        else if text.contains("llama") { 0.88 }
        else if text.contains("mistral") || text.contains("mixtral") { 0.87 }
        else if text.contains("gemma") { 0.82 }
        else if text.contains("phi") { 0.78 }
        else { 0.72 };
    let param_bonus = (candidate.parameter_billions.unwrap_or(3.0) / 14.0).min(1.0) * 0.18;
    (family * 0.82 + param_bonus).min(1.0)
}

fn usability_score(candidate: &Candidate) -> f64 {
    let text = format!("{} {} {}", candidate.repo_id, candidate.filename, candidate.tags.join(" ")).to_lowercase();
    let mut score: f64 = 0.35;
    if text.contains("instruct") || text.contains("chat") { score += 0.35; }
    if text.contains("license:") { score += 0.15; }
    if candidate.filename.to_lowercase().ends_with(".gguf") { score += 0.15; }
    score.min(1.0)
}

fn task_matches(candidate: &Candidate, task: &str) -> bool {
    let text = format!("{} {} {}", candidate.repo_id, candidate.filename, candidate.tags.join(" ")).to_lowercase();
    match task {
        "coding" => text.contains("coder") || text.contains("code") || text.contains("codestral"),
        "writing" => text.contains("llama") || text.contains("mistral") || text.contains("myth") || text.contains("story"),
        "research" => text.contains("deepseek") || text.contains("qwen") || text.contains("reason"),
        "small-business" => text.contains("instruct") || text.contains("chat"),
        _ => true,
    }
}

fn choose_runtime(hardware: &HardwareProfile, candidate: &Candidate, gpu_offload_realistic: bool) -> String {
    let text = format!("{} {}", candidate.repo_id, candidate.filename).to_lowercase();
    if text.contains("gguf") || candidate.filename.to_lowercase().ends_with(".gguf") {
        if gpu_offload_realistic { "llama.cpp with GPU offload".to_string() } else { "llama.cpp CPU mode or Ollama for ease".to_string() }
    } else if hardware.gpu_name.as_deref().unwrap_or_default().to_lowercase().contains("nvidia") {
        "vLLM or TensorRT-LLM".to_string()
    } else {
        "Ollama or LM Studio".to_string()
    }
}

fn runtime_recommendations(hardware: &HardwareProfile, candidate: &Candidate) -> Vec<String> {
    let mut runtimes = vec!["llama.cpp".to_string(), "Ollama".to_string(), "LM Studio".to_string()];
    let nvidia = hardware.gpu_name.as_deref().unwrap_or_default().to_lowercase().contains("nvidia");
    if nvidia && hardware.vram_gb.unwrap_or(0.0) >= 12.0 {
        runtimes.extend(["ExLlama".to_string(), "vLLM".to_string(), "TensorRT-LLM".to_string()]);
    }
    if candidate.parameter_billions.unwrap_or(7.0) <= 7.0 {
        runtimes.push("MLC".to_string());
    }
    runtimes
}

fn explain(candidate: &Candidate, hardware: &HardwareProfile, estimated_memory_gb: f64, fit_score: f64, quality_score: f64, speed_score: f64, gpu_offload_realistic: bool) -> (String, Vec<String>, Vec<String>) {
    let mut reasons = Vec::new();
    let mut cautions = Vec::new();
    reasons.push(format!("Estimated loaded memory footprint is about {:.1} GB, including runtime overhead.", estimated_memory_gb));
    if fit_score >= 0.7 {
        reasons.push(format!("It fits within the usable memory estimate for this machine ({:.1} GB available right now).", hardware.available_ram_gb));
    } else {
        cautions.push("This model may run, but the memory margin is tight. Expect paging, slow tokens/sec, or failed loads if other apps are open.".to_string());
    }
    if gpu_offload_realistic {
        reasons.push(format!("Detected VRAM appears sufficient for meaningful GPU offload ({:.1} GB VRAM).", hardware.vram_gb.unwrap_or(0.0)));
    } else if hardware.gpu_name.is_some() {
        cautions.push("GPU exists, but VRAM headroom does not look comfortable. CPU-heavy inference or partial offload is more realistic.".to_string());
    } else {
        cautions.push("No usable GPU was detected, so this is scored for CPU-friendly local inference.".to_string());
    }
    if let Some(q) = &candidate.quantization {
        reasons.push(format!("{q} quantization gives a practical balance of memory use, speed, and quality."));
    } else {
        cautions.push("Quantization could not be inferred from the filename, so the memory estimate is less certain.".to_string());
    }
    let decision = if fit_score < 0.4 { "Avoid unless you are willing to tune runtime settings." }
        else if speed_score > quality_score { "Best for fast, low-friction local chat." }
        else if quality_score > 0.82 { "Best quality option that still appears to fit." }
        else { "Balanced local recommendation." };
    (decision.to_string(), reasons, cautions)
}

fn extract_quantization(filename: &str) -> Option<String> {
    let regex = Regex::new(r"(?i)\b(Q[2-8](?:_[A-Z0-9]+)?|F16)\b").ok()?;
    regex.captures(filename).and_then(|captures| captures.get(1)).map(|m| m.as_str().to_uppercase())
}

fn extract_params(text: &str) -> Option<f64> {
    let regex = Regex::new(r"(?i)(?:^|[-_/.])([0-9]+(?:\.[0-9]+)?)\s*b(?:[-_/.]|$)").ok()?;
    regex.captures(text).and_then(|captures| captures.get(1)).and_then(|m| m.as_str().parse::<f64>().ok())
}

fn bytes_to_gb(value: u64) -> f64 { (value as f64 / 1024_f64.powi(3) * 10.0).round() / 10.0 }
fn round3(value: f64) -> f64 { (value * 1000.0).round() / 1000.0 }
fn sanitize(value: &str) -> String { value.replace('/', "__").replace('\\', "__").replace(':', "_") }

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![scan_hardware, recommend_models, download_model])
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
        Candidate { repo_id: name.into(), filename: file.into(), quantization: extract_quantization(file), parameter_billions: Some(params), size_gb: Some(size), downloads: 100000, likes: 200, tags: vec!["gguf".into(), "instruct".into()], updated_recently: true }
    }

    #[test]
    fn q4_model_scores_as_good_fit() {
        let req = RecommendationRequest { task: "general".into(), query: "GGUF instruct".into(), speed_quality_preference: 0.5, limit: 5 };
        let rec = score_candidate(&candidate("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct.Q4_K_M.gguf", 7.0, 4.8), &hardware(), &req);
        assert!(rec.fit_score > 0.7);
        assert!(rec.score > 0.6);
        assert!(rec.reasons.iter().any(|r| r.contains("memory")));
    }

    #[test]
    fn huge_model_is_penalized_for_fit() {
        let req = RecommendationRequest { task: "general".into(), query: "GGUF instruct".into(), speed_quality_preference: 0.9, limit: 5 };
        let rec = score_candidate(&candidate("Meta/Llama-70B-GGUF", "llama-70b.Q8_0.gguf", 70.0, 74.0), &hardware(), &req);
        assert!(rec.fit_score < 0.4);
        assert!(rec.cautions.iter().any(|c| c.contains("tight")));
    }
}
