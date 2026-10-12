# Tiny local web server for the ISL Avatar app (no installs needed). Close this window to stop the app.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$types = @{ '.html'='text/html; charset=utf-8'; '.js'='text/javascript; charset=utf-8'; '.mjs'='text/javascript; charset=utf-8';
            '.css'='text/css'; '.json'='application/json'; '.glb'='model/gltf-binary'; '.wasm'='application/wasm';
            '.png'='image/png'; '.jpg'='image/jpeg'; '.svg'='image/svg+xml'; '.txt'='text/plain'; '.task'='application/octet-stream' }
$listener = $null
foreach ($port in 8765..8775) {
  try { $l = New-Object System.Net.HttpListener; $l.Prefixes.Add("http://localhost:$port/"); $l.Start(); $listener = $l; break } catch { }
}
if (-not $listener) { Write-Host 'Could not open a local port (8765-8775). Close other copies of the app and try again.'; Read-Host 'Press Enter to exit'; exit 1 }
$url = "http://localhost:$port/"
Write-Host ''
Write-Host "  ISL Avatar app is running at $url"
Write-Host '  Use Google Chrome or Microsoft Edge (needed for speech input).'
Write-Host '  Keep this window open while you use the app. Close it to stop.'
Write-Host ''
$chrome = @("$env:ProgramFiles\Google\Chrome\Application\chrome.exe", "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe", "$env:LocalAppData\Google\Chrome\Application\chrome.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($chrome) { Start-Process $chrome $url } else { Start-Process "msedge" $url -ErrorAction SilentlyContinue; if (-not $?) { Start-Process $url } }
$rootFull = [System.IO.Path]::GetFullPath($root)
while ($listener.IsListening) {
  try {
    $ctx = $listener.GetContext(); $req = $ctx.Request; $res = $ctx.Response
    $rel = [Uri]::UnescapeDataString($req.Url.AbsolutePath.TrimStart('/')); if ($rel -eq '') { $rel = 'index.html' }
    $path = [System.IO.Path]::GetFullPath((Join-Path $rootFull $rel))
    if ($path.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path $path -PathType Leaf)) {
      $ext = [System.IO.Path]::GetExtension($path).ToLower()
      $res.ContentType = $(if ($types.ContainsKey($ext)) { $types[$ext] } else { 'application/octet-stream' })
      $res.Headers.Add('Cache-Control', 'no-cache')
      $bytes = [System.IO.File]::ReadAllBytes($path); $res.ContentLength64 = $bytes.Length; $res.OutputStream.Write($bytes, 0, $bytes.Length)
    } else { $res.StatusCode = 404 }
    $res.OutputStream.Close()
  } catch { }
}
