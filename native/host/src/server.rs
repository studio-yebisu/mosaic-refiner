use crate::manager::{self, atomic_json, Host, Runtime};
use percent_encoding::{percent_decode_str, utf8_percent_encode, NON_ALPHANUMERIC};
use serde_json::{json, Value};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
    thread,
    time::Duration,
};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

fn header<'a>(request: &'a Request, name: &'static str) -> Option<&'a str> {
    request
        .headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str())
}
fn security_headers<R: Read>(result: Response<R>) -> Response<R> {
    result
        .with_header(Header::from_bytes("X-Frame-Options", "DENY").unwrap())
        .with_header(
            Header::from_bytes("Content-Security-Policy", "frame-ancestors 'none'").unwrap(),
        )
        .with_header(Header::from_bytes("Cross-Origin-Resource-Policy", "same-origin").unwrap())
        .with_header(Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap())
        .with_header(Header::from_bytes("Referrer-Policy", "no-referrer").unwrap())
}
fn authorized(request: &Request, token: &str, media: bool) -> bool {
    let headers: Vec<_> = request
        .headers()
        .iter()
        .filter(|h| h.field.equiv("X-Mosaic-Token"))
        .collect();
    if headers.len() == 1 && headers[0].value.as_str() == token {
        return true;
    }
    if !media || !headers.is_empty() {
        return false;
    }
    let values: Vec<_> = request
        .url()
        .split_once('?')
        .map(|(_, q)| q)
        .unwrap_or("")
        .split('&')
        .filter_map(|p| p.strip_prefix("token="))
        .collect();
    values.len() == 1 && values[0] == token
}
fn local_source(path: &std::path::Path) -> bool {
    if !path.is_absolute() {
        return false;
    }
    #[cfg(windows)]
    {
        use std::path::{Component, Prefix};
        matches!(path.components().next(), Some(Component::Prefix(p)) if matches!(p.kind(), Prefix::Disk(_)))
    }
    #[cfg(not(windows))]
    {
        !path.to_string_lossy().starts_with("//")
    }
}
fn valid_origin(request: &Request, port: u16) -> bool {
    let mut origins = request.headers().iter().filter(|h| h.field.equiv("Origin"));
    let Some(origin) = origins.next() else {
        // Existing local CLI clients have no browser Origin header.
        return true;
    };
    origins.next().is_none()
        && [
            format!("http://127.0.0.1:{port}"),
            format!("http://localhost:{port}"),
        ]
        .iter()
        .any(|allowed| allowed == origin.value.as_str())
}
fn response(request: Request, code: u16, value: Value) {
    let _ = request.respond(security_headers(
        Response::from_string(value.to_string())
            .with_status_code(StatusCode(code))
            .with_header(
                Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap(),
            )
            .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap()),
    ));
}
fn source_suffix(path: &std::path::Path) -> Result<String, String> {
    let suffix = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if !["mp4", "mkv", "mov", "nut", "avi", "webm"].contains(&suffix.as_str()) {
        return Err("対応動画: MP4 / MKV / MOV / NUT / AVI / WebM".into());
    }
    Ok(suffix)
}
fn action(host: &Arc<Host>, path: &str, data: Value) -> Result<Value, String> {
    if !data.is_object() {
        return Err("設定はJSON objectが必要です".into());
    }
    match path {
        "/api/start" => host.start(data, false),
        "/api/resume" => host.start(data, true),
        "/api/cancel" => host.cancel(),
        "/api/history-select" => {
            host.select_history(data["job_id"].as_str().ok_or("job IDを指定してください")?)
        }
        "/api/history-cleanup" => {
            host.cleanup_history(data["job_id"].as_str().ok_or("job IDを指定してください")?)
        }
        "/api/local" => {
            let source = PathBuf::from(data["path"].as_str().unwrap_or(""));
            if !local_source(&source) {
                return Err("このPC内の動画の絶対パスを指定してください".into());
            }
            source_suffix(&source)?;
            if !source.is_file() {
                return Err("指定した動画を読み取れません".into());
            }
            host.reserve()?;
            let result = host.select(
                source.clone(),
                source.file_name().unwrap().to_string_lossy().into_owned(),
                false,
            );
            host.release();
            result
        }
        "/api/show-preview" => {
            host.reserve()?;
            let result = (|| {
                let mut session = host.state.lock().unwrap().session.clone();
                if session.is_null() {
                    return Err("動画を先に選択してください".into());
                }
                let target = host
                    .directory
                    .join("previews")
                    .join(session["id"].as_str().ok_or("idが不正です")?)
                    .join("preview.png");
                host.runtime.rpc(
                    "preview",
                    json!({"source":session["spec"]["source"],"target":target}),
                )?;
                session["preview"] = json!(target);
                session["preview_enabled"] = json!(true);
                atomic_json(&host.directory.join("session.json"), &session)?;
                host.state.lock().unwrap().session = session;
                Ok(host.snapshot())
            })();
            host.release();
            result
        }
        "/api/prepare-detector" | "/api/prepare-restoration" => {
            host.reserve()?;
            let result =
                host.runtime
                    .python_worker(path.trim_start_matches("/api/"), data, |event| {
                        let mut state = host.state.lock().unwrap();
                        if let Some(phase) = event["phase"].as_str() {
                            let model = event["model"].as_str().unwrap_or("");
                            let bytes =
                                match (event["downloaded"].as_u64(), event["total"].as_u64()) {
                                    (Some(done), Some(total)) => format!(
                                        " · {:.1} / {:.1} MiB",
                                        done as f64 / 1048576.0,
                                        total as f64 / 1048576.0
                                    ),
                                    _ => String::new(),
                                };
                            state.value["phase"] = json!(format!("{phase} / {model}{bytes}"));
                        }
                    });
            if let Ok(meta) = &result {
                let mut state = host.state.lock().unwrap();
                state.metadata = meta.clone();
                state.value["phase"] = json!("補助モデルの準備を確認しました");
            }
            host.release();
            result.map(|_| host.snapshot())
        }
        "/api/shutdown" => {
            host.stopping.store(true, Ordering::SeqCst);
            Ok(json!({"stopping":true}))
        }
        _ => Err("操作が見つかりません".into()),
    }
}
fn upload(host: &Arc<Host>, request: &mut Request) -> Result<Value, String> {
    let length = request.body_length().ok_or("動画のbyte数が必要です")?;
    if length == 0 {
        return Err("空の動画は読み込めません".into());
    }
    let name = percent_decode_str(header(request, "X-Filename").unwrap_or("video.mp4"))
        .decode_utf8()
        .map_err(|e| e.to_string())?
        .replace('\\', "/")
        .rsplit('/')
        .next()
        .unwrap_or("video.mp4")
        .to_owned();
    let suffix = source_suffix(&PathBuf::from(&name))?;
    host.reserve()?;
    let folder = host.directory.join("inputs").join(manager::identifier());
    let partial = folder.join("upload.partial");
    let result = (|| {
        fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        if fs2::available_space(&folder).map_err(|e| e.to_string())?
            < (length as u64).saturating_add(256 * 1024 * 1024)
        {
            return Err("動画の保存に必要な空き容量がありません".into());
        }
        let mut target = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .map_err(|e| e.to_string())?;
        let copied = std::io::copy(&mut request.as_reader().take(length as u64), &mut target)
            .map_err(|e| e.to_string())?;
        if copied != length as u64 {
            return Err("動画の転送が途中で切れました".into());
        }
        target.flush().map_err(|e| e.to_string())?;
        target.sync_all().map_err(|e| e.to_string())?;
        drop(target);
        let source = folder.join(format!("source.{suffix}"));
        fs::rename(&partial, &source).map_err(|e| e.to_string())?;
        host.select(source, name, true)
    })();
    let _ = fs::remove_file(&partial);
    host.release();
    result
}
fn handle(mut request: Request, host: &Arc<Host>, token: &str, port: u16) {
    let allowed = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    if !header(&request, "Host").is_some_and(|h| allowed.iter().any(|v| v == h)) {
        response(request, 403, json!({"error":"localhostから開いてください"}));
        return;
    }
    let path = request.url().split('?').next().unwrap_or("").to_owned();
    let media =
        request.method() == &Method::Get && (path == "/api/preview" || path == "/api/output");
    if path.starts_with("/api/") && !authorized(&request, token, media) {
        response(
            request,
            403,
            json!({"error":"start-web.cmdからWebGUIを開き直してください"}),
        );
        return;
    }
    if request.method() == &Method::Get {
        if path == "/" {
            let page = include_str!("../../../mosaic_refiner/web.html").replace(
                "__PROCESSING_PRESETS__",
                &host.snapshot()["processing_presets"].to_string(),
            );
            let _ = request.respond(security_headers(
                Response::from_string(page)
                    .with_header(
                        Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap(),
                    )
                    .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap()),
            ));
            return;
        }
        if path == "/api/status" {
            response(request, 200, host.snapshot());
            return;
        }
        if path == "/api/history" {
            match host.history() {
                Ok(value) => response(request, 200, value),
                Err(error) => response(request, 400, json!({"error":error})),
            }
            return;
        }
        if path == "/api/preview" || path == "/api/output" {
            let snapshot = host.snapshot();
            let session = &snapshot["session"];
            if session.is_null() {
                response(request, 404, json!({"error":"動画が選択されていません"}));
                return;
            }
            let preview = path.ends_with("preview");
            if preview && session["preview_enabled"] != true {
                response(request, 409, json!({"error":"プレビューは非表示です"}));
                return;
            }
            if !preview && snapshot["state"] != "complete" {
                response(request, 409, json!({"error":"出力はまだ完成していません"}));
                return;
            }
            let path = if preview {
                &session["preview"]
            } else {
                &session["spec"]["output"]
            };
            let Some(path) = path.as_str() else {
                response(request, 404, json!({"error":"ファイルがありません"}));
                return;
            };
            match File::open(path) {
                Ok(file) => {
                    let mut result = Response::from_file(file)
                        .with_header(
                            Header::from_bytes(
                                "Content-Type",
                                if preview {
                                    "image/png"
                                } else {
                                    "application/octet-stream"
                                },
                            )
                            .unwrap(),
                        )
                        .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap());
                    if !preview {
                        let name = format!(
                            "{}.refined.{}",
                            PathBuf::from(session["name"].as_str().unwrap_or("video"))
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy(),
                            PathBuf::from(path)
                                .extension()
                                .unwrap_or_default()
                                .to_string_lossy()
                        );
                        result = result.with_header(
                            Header::from_bytes(
                                "Content-Disposition",
                                format!(
                                    "attachment; filename*=UTF-8''{}",
                                    utf8_percent_encode(&name, NON_ALPHANUMERIC)
                                ),
                            )
                            .unwrap(),
                        );
                    }
                    let _ = request.respond(security_headers(result));
                }
                Err(_) => response(request, 404, json!({"error":"ファイルが見つかりません"})),
            }
            return;
        }
        response(request, 404, json!({"error":"見つかりません"}));
        return;
    }
    if request.method() != &Method::Post {
        response(request, 405, json!({"error":"GET/POSTが必要です"}));
        return;
    }
    if !authorized(&request, token, false) {
        response(request, 403, json!({"error":"WebGUIを開き直してください"}));
        return;
    }
    if !valid_origin(&request, port) {
        response(
            request,
            403,
            json!({"error":"このPCのWebGUIから操作してください"}),
        );
        return;
    }
    let result = if path == "/api/upload" {
        upload(host, &mut request)
    } else {
        (|| {
            let length = request.body_length().ok_or("Content-Lengthが必要です")?;
            if length > 65536 {
                return Err("設定リクエストが大きすぎます".into());
            }
            let mut bytes = Vec::with_capacity(length);
            request
                .as_reader()
                .take(65537)
                .read_to_end(&mut bytes)
                .map_err(|e| e.to_string())?;
            if bytes.len() != length {
                return Err("設定リクエストが途中で切れました".into());
            }
            let data = if bytes.is_empty() {
                json!({})
            } else {
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?
            };
            action(host, &path, data)
        })()
    };
    match result {
        Ok(value) => response(request, 200, value),
        Err(error) => response(request, 400, json!({"error":error})),
    }
}
pub fn serve(
    runtime: Runtime,
    directory: PathBuf,
    port: u16,
    backend: String,
    open_browser: bool,
) -> Result<i32, String> {
    let stopping = manager::install_stop_handler()?;
    let host = Host::new(
        manager::absolute(&directory)?,
        runtime,
        backend,
        stopping.clone(),
    )?;
    let server = Arc::new(Server::http(("127.0.0.1", port)).map_err(|e| e.to_string())?);
    let port = server
        .server_addr()
        .to_ip()
        .ok_or("loopback socketがありません")?
        .port();
    let token = format!("{}{}", manager::identifier(), manager::identifier());
    let url = format!("http://127.0.0.1:{port}/#token={token}");
    println!("WebGUI: {url} (Rust)");
    if open_browser {
        #[cfg(windows)]
        {
            let _ = std::process::Command::new("rundll32.exe")
                .args(["url.dll,FileProtocolHandler", &url])
                .spawn();
        }
        #[cfg(not(windows))]
        {
            let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
        }
    }
    let mut workers = Vec::new();
    for _ in 0..8 {
        let server = server.clone();
        let host = host.clone();
        let token = token.clone();
        let stopping = stopping.clone();
        workers.push(thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                match server.recv_timeout(Duration::from_millis(200)) {
                    Ok(Some(request)) => handle(request, &host, &token, port),
                    Ok(None) => {}
                    Err(_) => break,
                }
            }
        }));
    }
    for worker in workers {
        let _ = worker.join();
    }
    let _ = host.cancel();
    while host.state.lock().unwrap().busy {
        thread::sleep(Duration::from_millis(100));
    }
    Ok(0)
}
