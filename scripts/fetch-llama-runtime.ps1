$ErrorActionPreference = 'Stop'

$runtimeDir = Join-Path $PSScriptRoot '..\src-tauri\resources\runtime'
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

Write-Host 'Resolving the latest official llama.cpp Windows release...'
$release = Invoke-RestMethod -Headers @{ 'User-Agent' = 'MyAILocalModel-Build' } -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest'

$preferredPatterns = @(
  'bin-win-vulkan-x64\.zip$',
  'bin-win-cpu-x64\.zip$',
  'bin-win-avx2-x64\.zip$'
)

$asset = $null
foreach ($pattern in $preferredPatterns) {
  $asset = $release.assets | Where-Object { $_.name -match $pattern } | Select-Object -First 1
  if ($asset) { break }
}

if (-not $asset) {
  $available = ($release.assets | ForEach-Object { $_.name }) -join "`n"
  throw "Could not find a supported llama.cpp Windows x64 archive. Available assets:`n$available"
}

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('myailocalmodel-llama-' + [Guid]::NewGuid())
$zipPath = Join-Path $tempRoot $asset.name
$extractPath = Join-Path $tempRoot 'extract'
New-Item -ItemType Directory -Force -Path $extractPath | Out-Null

Write-Host "Downloading $($asset.name)..."
Invoke-WebRequest -Headers @{ 'User-Agent' = 'MyAILocalModel-Build' } -Uri $asset.browser_download_url -OutFile $zipPath
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

$server = Get-ChildItem -Path $extractPath -Filter 'llama-server.exe' -Recurse | Select-Object -First 1
if (-not $server) { throw 'The downloaded llama.cpp archive did not contain llama-server.exe.' }

Remove-Item -Path (Join-Path $runtimeDir '*') -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $server.Directory.FullName '*') -Destination $runtimeDir -Recurse -Force

$metadata = [ordered]@{
  release = $release.tag_name
  asset = $asset.name
  source = $asset.browser_download_url
  fetchedUtc = [DateTime]::UtcNow.ToString('o')
}
$metadata | ConvertTo-Json | Set-Content -Path (Join-Path $runtimeDir 'runtime-manifest.json') -Encoding UTF8

Write-Host "Bundled llama.cpp $($release.tag_name) from $($asset.name)."
Remove-Item -Path $tempRoot -Recurse -Force
