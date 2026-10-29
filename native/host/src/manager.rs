use fs2::FileExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
pub fn identifier() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}
pub fn sidecar(job: &Path, suffix: &str) -> PathBuf {
    let mut path = job.as_os_str().to_os_string();
    path.push(".");
    path.push(suffix);
    PathBuf::from(path)
}
pub fn migrate_legacy_spec(job: &Path) -> Result<(), String> {
    let current = sidecar(job, "host-spec.json");
    let legacy = job.with_extension("host-spec.json");
    if current == legacy {
        return Ok(());
    }
    let owner = job.with_extension("host-owner.lock");
    if owner.exists()
        && OpenOptions::new()
            .read(true)
            .write(true)
            .open(&owner)
            .map_or(true, |file| file.try_lock_exclusive().is_err())
    {
        return Err("旧版のdot付きjobが実行中です。保存・終了を待ってください".into());
    }
    if !current.exists() && legacy.exists() {
        let spec = read_json(&legacy)?;
        let saved = PathBuf::from(
            spec["job_dir"]
                .as_str()
                .ok_or("旧job記録の保存先が不正です")?,
        );
        if absolute(&saved)? != absolute(job)? {
            return Err(
                "旧dot付きjobの記録が別jobと衝突しています。共有記録を保全して確認してください"
                    .into(),
            );
        }
        atomic_json(&current, &spec)?;
    }
    Ok(())
}
pub fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_reader(BufReader::new(File::open(path).map_err(|e| e.to_string())?))
        .map_err(|e| e.to_string())
}
pub fn atomic_json(path: &Path, value: &Value) -> Result<(), String> {
    let parent = path.parent().ok_or("JSONのparentがありません")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    {
        let mut writer = BufWriter::new(&mut file);
        serde_json::to_writer_pretty(&mut writer, value).map_err(|e| e.to_string())?;
        writer.write_all(b"\n").map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
    }
    file.as_file().sync_all().map_err(|e| e.to_string())?;
    file.persist(path).map_err(|e| e.to_string())?;
    Ok(())
}
pub fn hash(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
pub fn absolute(path: &Path) -> Result<PathBuf, String> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    // Canonical Windows paths retain the normal prefix expected by the Python
    // manifests, not a new verbatim \\?\ spelling on every launch.
    Ok(PathBuf::from(
        path.to_string_lossy().trim_start_matches("\\\\?\\"),
    ))
}
pub fn summary(directory: &Path) -> Result<Value, String> {
    Ok(saved_record(directory)?.summary())
}
#[derive(serde::Deserialize)]
struct SavedRecord {
    state: String,
    job_id: String,
    units: Vec<SavedUnit>,
    #[serde(default)]
    stage: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    output: Option<String>,
    #[serde(default)]
    output_sha256: Option<String>,
    #[serde(default)]
    last_error: Value,
    #[serde(default)]
    verification: Value,
    #[serde(default)]
    config: SavedConfig,
}
#[derive(serde::Deserialize)]
struct SavedUnit {
    #[serde(default)]
    state: Option<String>,
}
#[derive(serde::Deserialize, Default)]
struct SavedConfig {
    #[serde(default)]
    source_sha256: Option<String>,
}
fn saved_record(directory: &Path) -> Result<SavedRecord, String> {
    serde_json::from_reader(BufReader::new(
        File::open(directory.join("manifest.json")).map_err(|e| e.to_string())?,
    ))
    .map_err(|e| e.to_string())
}
impl SavedRecord {
    fn summary(&self) -> Value {
        json!({"stage":self.stage.as_deref().unwrap_or("cpu-diagnostic-no-ai"),"state":self.state,"job_id":self.job_id,
            "completed_units":self.units.iter().filter(|u|u.state.as_deref()==Some("complete")).count(),"total_units":self.units.len(),
            "output":self.output,"last_error":self.last_error,"verification":self.verification,"integrity_checked":false})
    }
}
pub fn parse_roi(text: &str) -> Result<Value, String> {
    let values = text
        .split(',')
        .map(|s| s.parse::<i64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "ROIは整数4個です")?;
    if values.len() != 4 {
        return Err("ROIはx,y,width,heightです".into());
    }
    Ok(json!({"x":values[0],"y":values[1],"width":values[2],"height":values[3]}))
}
pub fn install_stop_handler() -> Result<Arc<AtomicBool>, String> {
    let stop = Arc::new(AtomicBool::new(false));
    let signal = stop.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::SeqCst)).map_err(|e| e.to_string())?;
    Ok(stop)
}
#[derive(Clone)]
pub struct Runtime {
    pub root: PathBuf,
    pub python: PathBuf,
}

// Only bounded, one-shot operations own a process tree. Inference and CLI jobs
// keep their existing save-at-safe-point cancellation, without a wall deadline.
fn worker_timeout(operation: &str) -> Option<Duration> {
    let seconds = match operation {
        "metadata" | "options" => 30,
        "probe" | "preview" => 120,
        "cleanup" => 1800,
        "prepare-detector" | "prepare-restoration" => 3600,
        _ => return None,
    };
    Some(Duration::from_secs(seconds))
}

#[cfg(windows)]
mod worker_job {
    use std::{ffi::c_void, io, mem, os::windows::io::AsRawHandle, process::Child, ptr};

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimits {
        process_time: i64,
        job_time: i64,
        flags: u32,
        minimum_working_set: usize,
        maximum_working_set: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }
    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimits {
        basic: BasicLimits,
        io_counters: [u64; 6],
        process_memory: usize,
        job_memory: usize,
        peak_process_memory: usize,
        peak_job_memory: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attributes: *const c_void, name: *const u16) -> *mut c_void;
        fn SetInformationJobObject(
            job: *mut c_void,
            class: i32,
            data: *const c_void,
            length: u32,
        ) -> i32;
        fn AssignProcessToJobObject(job: *mut c_void, process: *mut c_void) -> i32;
        fn TerminateJobObject(job: *mut c_void, code: u32) -> i32;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }
    // Native signature: phnt/include/ntpsapi.h in winsiderss/systeminformer.
    // The process is created suspended so a venv launcher cannot create a
    // descendant before the job has been assigned.
    #[link(name = "ntdll")]
    extern "system" {
        fn NtResumeProcess(process: *mut c_void) -> i32;
    }
    pub struct Job(*mut c_void);
    impl Job {
        pub fn assign(child: &Child) -> io::Result<Self> {
            let job = Self(unsafe { CreateJobObjectW(ptr::null(), ptr::null()) });
            if job.0.is_null() {
                return Err(io::Error::last_os_error());
            }
            let mut limits = ExtendedLimits::default();
            limits.basic.flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if unsafe {
                SetInformationJobObject(
                    job.0,
                    9,
                    &limits as *const _ as *const c_void,
                    mem::size_of::<ExtendedLimits>() as u32,
                )
            } == 0
                || unsafe { AssignProcessToJobObject(job.0, child.as_raw_handle()) } == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }
        pub fn terminate(&self) {
            unsafe {
                TerminateJobObject(self.0, 1);
            }
        }
        pub fn resume(child: &Child) -> io::Result<()> {
            let status = unsafe { NtResumeProcess(child.as_raw_handle()) };
            if status < 0 {
                return Err(io::Error::other(format!(
                    "NtResumeProcess failed: NTSTATUS 0x{:08X}",
                    status as u32
                )));
            }
            Ok(())
        }
    }
    impl Drop for Job {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }
}

struct WorkerProcess {
    child: Child,
    #[cfg(unix)]
    bounded: bool,
    #[cfg(windows)]
    job: Option<worker_job::Job>,
}
impl WorkerProcess {
    fn spawn(command: &mut Command, bounded: bool) -> Result<Self, String> {
        #[cfg(windows)]
        if bounded {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000204); // NO_WINDOW | NEW_PROCESS_GROUP | SUSPENDED
        }
        #[cfg(unix)]
        if bounded {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command.spawn().map_err(|e| e.to_string())?;
        #[cfg(windows)]
        let job = if bounded {
            // The trusted worker reads stdin before launching FFmpeg. Assign
            // it before sending the request so descendants inherit this job.
            match worker_job::Job::assign(&child) {
                Ok(job) => {
                    if let Err(error) = worker_job::Job::resume(&child) {
                        job.terminate();
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(format!("workerを再開できません: {error}"));
                    }
                    Some(job)
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!("workerの実行範囲を準備できません: {error}"));
                }
            }
        } else {
            None
        };
        Ok(Self {
            child,
            #[cfg(unix)]
            bounded,
            #[cfg(windows)]
            job,
        })
    }
    fn terminate(&mut self) {
        #[cfg(windows)]
        if let Some(job) = &self.job {
            job.terminate();
        }
        #[cfg(unix)]
        if self.bounded {
            extern "C" {
                fn kill(pid: i32, signal: i32) -> i32;
            }
            unsafe {
                kill(-(self.child.id() as i32), 9);
            }
        }
        let _ = self.child.kill();
    }
}
impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            self.terminate();
            let _ = self.child.wait();
        }
    }
}

struct CancellationWatcher {
    done: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Drop for CancellationWatcher {
    fn drop(&mut self) {
        self.done.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

enum WorkerEvent {
    Input(Result<(), String>),
    Line(Result<String, String>),
    Closed,
}

fn communicate_worker<F: FnMut(Value)>(
    child: &mut WorkerProcess,
    operation: &str,
    request: Vec<u8>,
    data: &Value,
    mut progress: F,
    stop: Option<Arc<AtomicBool>>,
    timeout: Option<Duration>,
) -> Result<Value, String> {
    let mut deadline = timeout.map(|limit| Instant::now() + limit);
    let progress_extends_deadline = matches!(operation, "prepare-detector" | "prepare-restoration");
    let timeout_error = || {
        format!("workerが時間内に完了しませんでした（{operation} / {}秒）。入力や接続先を確認して再試行してください", timeout.unwrap().as_secs_f64())
    };
    let (send, receive) = mpsc::sync_channel(32);
    let mut input = child.child.stdin.take().ok_or("workerの入力がありません")?;
    let output = child
        .child
        .stdout
        .take()
        .ok_or("workerの出力がありません")?;
    let sender = send.clone();
    let writer = thread::spawn(move || {
        let result = input.write_all(&request).map_err(|e| e.to_string());
        drop(input);
        let _ = sender.send(WorkerEvent::Input(result));
    });
    let reader = thread::spawn(move || {
        for line in BufReader::new(output).lines() {
            if send
                .send(WorkerEvent::Line(line.map_err(|e| e.to_string())))
                .is_err()
            {
                return;
            }
        }
        let _ = send.send(WorkerEvent::Closed);
    });
    let done = Arc::new(AtomicBool::new(false));
    let watcher = CancellationWatcher {
        done: done.clone(),
        thread: stop.map(|stop| {
            let job = data["job_dir"].as_str().map(PathBuf::from);
            thread::spawn(move || {
                while !done.load(Ordering::SeqCst) {
                    if stop.load(Ordering::SeqCst) {
                        if let Some(job) = &job {
                            let _ = atomic_json(
                                &sidecar(job, "host-cancel.json"),
                                &json!({"cancel":true}),
                            );
                            if job.join("manifest.json").exists() {
                                let _ =
                                    atomic_json(&job.join("cancel.json"), &json!({"cancel":true}));
                            }
                        }
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
            })
        }),
    };
    let mut result = None;
    let mut error = None;
    let mut terminate = false;
    loop {
        if deadline.is_some_and(|end| Instant::now() >= end) {
            error = Some(timeout_error());
            terminate = true;
            break;
        }
        match receive.recv_timeout(Duration::from_millis(50)) {
            Ok(WorkerEvent::Input(Err(message))) => {
                error = Some(message);
                terminate = true;
                break;
            }
            Ok(WorkerEvent::Input(Ok(()))) => {}
            Ok(WorkerEvent::Closed) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Ok(WorkerEvent::Line(Err(message))) => {
                error = Some(format!("worker出力を読み取れません: {message}"));
                if timeout.is_some() {
                    terminate = true;
                    break;
                }
            }
            Ok(WorkerEvent::Line(Ok(line))) => match serde_json::from_str::<Value>(&line) {
                Ok(event) => match event["kind"].as_str() {
                    Some("progress") if event["data"].is_object() => {
                        if progress_extends_deadline {
                            deadline = timeout.map(|limit| Instant::now() + limit);
                        }
                        progress(event["data"].clone())
                    }
                    Some("result") => result = Some(event["data"].clone()),
                    Some("error") => {
                        error = Some(
                            event["data"]["error"]
                                .as_str()
                                .unwrap_or("worker error")
                                .to_owned(),
                        )
                    }
                    _ => error = Some("worker protocolが不正です".into()),
                },
                Err(_) => error = Some("worker eventのJSONが不正です".into()),
            },
        }
    }
    if terminate {
        child.terminate();
    }
    let status = loop {
        match child.child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None => {
                if deadline.is_some_and(|end| Instant::now() >= end) {
                    error = Some(timeout_error());
                    child.terminate();
                    break child.child.wait().map_err(|e| e.to_string())?;
                }
                thread::sleep(Duration::from_millis(20));
            }
        }
    };
    // Drop the receiver before joining so a full bounded channel cannot strand
    // the reader after a timeout or protocol error.
    drop(receive);
    let _ = writer.join();
    let _ = reader.join();
    drop(watcher);
    if let Some(error) = error {
        return Err(error);
    }
    if !status.success() {
        return Err(format!("workerが異常終了しました: {status}"));
    }
    result.ok_or_else(|| "workerが結果を返しませんでした".into())
}
impl Runtime {
    pub fn validate(&self) -> Result<(), String> {
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let record = read_json(
            &executable
                .parent()
                .ok_or("host pathが不正です")?
                .join("host-build.json"),
        )?;
        if record["host_sha256"] != hash(&executable)? {
            return Err("Rust hostのhashが構築記録と異なります".into());
        }
        let mut files = std::collections::BTreeMap::<String, String>::new();
        fn walk(
            root: &Path,
            directory: &Path,
            files: &mut std::collections::BTreeMap<String, String>,
        ) -> Result<(), String> {
            for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
                let path = entry.map_err(|e| e.to_string())?.path();
                if path.file_name().is_some_and(|s| s == "target") {
                    continue;
                }
                if path.is_dir() {
                    walk(root, &path, files)?;
                } else if path.extension().is_some_and(|s| s == "rs")
                    || path.file_name().is_some_and(|s| {
                        s == "Cargo.toml" || s == "Cargo.lock" || s == "cli-schema.json"
                    })
                {
                    let text = fs::read(&path).map_err(|e| e.to_string())?;
                    let normalized = String::from_utf8(text)
                        .map_err(|e| e.to_string())?
                        .replace("\r\n", "\n");
                    files.insert(
                        path.strip_prefix(root)
                            .map_err(|e| e.to_string())?
                            .to_string_lossy()
                            .replace('\\', "/"),
                        format!("{:x}", Sha256::digest(normalized.as_bytes())),
                    );
                }
            }
            Ok(())
        }
        walk(&self.root, &self.root.join("native"), &mut files)?;
        let digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&files).map_err(|e| e.to_string())?)
        );
        let ui = fs::read_to_string(self.root.join("mosaic_refiner/web.html"))
            .map_err(|e| e.to_string())?
            .replace("\r\n", "\n");
        if record["source_sha256"] != digest
            || record["ui_sha256"] != format!("{:x}", Sha256::digest(ui.as_bytes()))
        {
            return Err(
                "Rust hostのsource/UI版が構築記録と異なります。明示的に再構築してください".into(),
            );
        }
        Ok(())
    }
    pub fn execute<F: FnMut(Value)>(
        &self,
        spec: &Value,
        progress: F,
        stop: Option<Arc<AtomicBool>>,
    ) -> Result<Value, String> {
        self.worker("execute", spec.clone(), progress, stop)
    }
    pub fn rpc(&self, operation: &str, data: Value) -> Result<Value, String> {
        self.worker(operation, data, |_| {}, None)
    }
    pub fn supervised_cli(&self, data: Value, stop: Arc<AtomicBool>) -> Result<Value, String> {
        self.launch(
            "cli",
            data,
            |event| eprintln!("{}", event),
            Some(stop),
            true,
        )
    }
    pub fn python_worker<F: FnMut(Value)>(
        &self,
        operation: &str,
        data: Value,
        progress: F,
    ) -> Result<Value, String> {
        self.launch(operation, data, progress, None, false)
    }
    fn worker<F: FnMut(Value)>(
        &self,
        operation: &str,
        data: Value,
        mut progress: F,
        stop: Option<Arc<AtomicBool>>,
    ) -> Result<Value, String> {
        self.launch(operation, data, &mut progress, stop, operation == "execute")
    }
    fn launch<F: FnMut(Value)>(
        &self,
        operation: &str,
        data: Value,
        mut progress: F,
        stop: Option<Arc<AtomicBool>>,
        native: bool,
    ) -> Result<Value, String> {
        let mut command = if native {
            let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
            command
                .arg("--project-root")
                .arg(&self.root)
                .arg("--worker-python")
                .arg(&self.python)
                .arg("_worker");
            command
        } else {
            let mut command = Command::new(&self.python);
            command.args(["-B", "-m", "mosaic_refiner.host_worker"]);
            command
        };
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000200);
        }
        command
            .current_dir(&self.root)
            .env("PYTHONUTF8", "1")
            .env("MOSAIC_REFINER_PYTHON_WORKER", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let timeout = worker_timeout(operation);
        let request = serde_json::to_vec(&json!({"operation":operation,"data":data}))
            .map_err(|e| e.to_string())?;
        let mut child = WorkerProcess::spawn(&mut command, timeout.is_some())?;
        communicate_worker(
            &mut child,
            operation,
            request,
            &data,
            &mut progress,
            stop,
            timeout,
        )
    }
}

pub struct State {
    pub value: Value,
    pub session: Value,
    pub metadata: Value,
    pub busy: bool,
    pub uploading: bool,
}
pub struct Host {
    pub directory: PathBuf,
    pub runtime: Runtime,
    pub backend: String,
    pub state: Mutex<State>,
    pub stopping: Arc<AtomicBool>,
    _lock: File,
}
impl Host {
    pub fn new(
        directory: PathBuf,
        runtime: Runtime,
        backend: String,
        stopping: Arc<AtomicBool>,
    ) -> Result<Arc<Self>, String> {
        fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.join(".host.lock"))
            .map_err(|e| e.to_string())?;
        lock.try_lock_exclusive()
            .map_err(|_| "このdirectoryは別のアプリが管理中です")?;
        let recent = directory.join("session.json");
        let session = if recent.exists() {
            read_json(&recent)?
        } else {
            Value::Null
        };
        let value = if session.is_null() {
            json!({"state":"idle","phase":"動画を選択してください"})
        } else {
            json!({"state":"ready","phase":"保存済みの条件で再開できます"})
        };
        let metadata = runtime.rpc("metadata", json!({}))?;
        let host = Arc::new(Self {
            directory,
            runtime,
            backend,
            state: Mutex::new(State {
                value,
                session,
                metadata,
                busy: false,
                uploading: false,
            }),
            stopping,
            _lock: lock,
        });
        let session = host.state.lock().unwrap().session.clone();
        if let Some(result) = completed(&session)? {
            host.state.lock().unwrap().value =
                json!({"state":"complete","phase":"完成動画を保存できます","result":result});
        }
        Ok(host)
    }
    pub fn snapshot(&self) -> Value {
        let state = self.state.lock().unwrap();
        let mut result = state.value.clone();
        for (key, value) in state.metadata.as_object().unwrap() {
            result[key] = value.clone();
        }
        result["session"] = state.session.clone();
        result["can_resume"] =
            json!(state.session["spec"]["job_dir"]
                .as_str()
                .is_some_and(|job| {
                    sidecar(Path::new(job), "host-spec.json").is_file()
                        || Path::new(job).join("manifest.json").is_file()
                }));
        result["busy"] = json!(state.busy);
        result["uploading"] = json!(state.uploading);
        result["server_time"] = json!(now());
        result["pipeline_revision"] = json!("rust-host-v2-model-auth");
        result["application_engine"] = json!("rust");
        result["application_root"] = json!(self.runtime.root);
        result["application_build"] =
            json!(option_env!("MOSAIC_NATIVE_SOURCE_SHA").unwrap_or("unrecorded"));
        result
    }
    pub fn history(&self) -> Result<Value, String> {
        crate::history::list(self)
    }
    pub fn select_history(&self, id: &str) -> Result<Value, String> {
        self.reserve()?;
        let result = (|| {
            let session = crate::history::session(self, id)?;
            let result = completed(&session)?;
            atomic_json(&self.directory.join("session.json"), &session)?;
            let mut state = self.state.lock().unwrap();
            state.session = session;
            state.value = match result {
                Some(result) => {
                    json!({"state":"complete","phase":"完成動画を保存できます","result":result})
                }
                None => {
                    json!({"state":"ready","phase":"履歴を読み込みました。「前の処理を再開」で保存済みの区切りから続けます"})
                }
            };
            drop(state);
            Ok(self.snapshot())
        })();
        self.release();
        result
    }
    pub fn cleanup_history(&self, id: &str) -> Result<Value, String> {
        self.reserve()?;
        let result = crate::history::cleanup(self, id);
        self.release();
        result.map(|cleanup| json!({"cleanup":cleanup}))
    }
    pub fn reserve(&self) -> Result<(), String> {
        let mut state = self.state.lock().unwrap();
        if state.busy || state.uploading || self.stopping.load(Ordering::SeqCst) {
            return Err("現在の処理が終わってから操作してください".into());
        }
        state.uploading = true;
        Ok(())
    }
    pub fn release(&self) {
        self.state.lock().unwrap().uploading = false;
    }
    pub fn select(&self, source: PathBuf, name: String, copied: bool) -> Result<Value, String> {
        let source = absolute(&source)?;
        let meta = self.runtime.rpc("probe", json!({"source":source}))?;
        let spec = new_spec(&source, &self.directory, &self.backend)?;
        let session = json!({"spec":spec,"name":name,"width":meta["width"],"height":meta["height"],"duration":meta["duration"],
            "preview":null,"preview_enabled":false,"copied_input":copied,"id":identifier()});
        atomic_json(&self.directory.join("session.json"), &session)?;
        {
            let mut state = self.state.lock().unwrap();
            state.session = session;
            state.value = json!({"state":"ready","phase":"読み込み完了。自動処理を開始できます"});
        }
        Ok(self.snapshot())
    }
    pub fn start(self: &Arc<Self>, settings: Value, resume: bool) -> Result<Value, String> {
        self.reserve()?;
        let result = self.start_reserved(settings, resume);
        self.release();
        result
    }
    fn start_reserved(self: &Arc<Self>, settings: Value, resume: bool) -> Result<Value, String> {
        let mut session = self.state.lock().unwrap().session.clone();
        if session.is_null() {
            return Err("動画を先に選択してください".into());
        }
        if resume {
            let job = PathBuf::from(
                session["spec"]["job_dir"]
                    .as_str()
                    .ok_or("job_dirが不正です")?,
            );
            if !sidecar(&job, "host-spec.json").is_file() && !job.join("manifest.json").is_file() {
                return Err("このjobはまだ開始していません。「補正を開始」を使ってください".into());
            }
            if let Some(result) = completed(&session)? {
                self.state.lock().unwrap().value =
                    json!({"state":"complete","phase":"完成動画を保存できます","result":result});
                return Ok(self.snapshot());
            }
            // A legacy unfinished job may share another job's output name.
            // Keep its verified units, source and settings; change only the destination.
            let output = PathBuf::from(
                session["spec"]["output"]
                    .as_str()
                    .ok_or("outputが不正です")?,
            );
            if output.exists() {
                session["spec"]["output"] = json!(unique_output(
                    &PathBuf::from(
                        session["spec"]["source"]
                            .as_str()
                            .ok_or("sourceが不正です")?
                    ),
                    job.file_name()
                        .and_then(|s| s.to_str())
                        .ok_or("job IDが不正です")?
                )?);
            }
        } else {
            let context = settings
                .get("context_seconds")
                .cloned()
                .unwrap_or(json!("auto"));
            if context != json!("auto") && ![json!(5), json!(10), json!(15)].contains(&context) {
                return Err("contextは自動・5・10・15秒です".into());
            }
            let roi = settings.get("roi").cloned().unwrap_or(Value::Null);
            validate_roi(
                &roi,
                session["width"].as_i64().ok_or("widthが不正です")?,
                session["height"].as_i64().ok_or("heightが不正です")?,
            )?;
            let options=self.runtime.rpc("options",json!({"processing_options":settings.get("processing_options").unwrap_or(&session["spec"]["processing_options"])}))?;
            let previous = &session["spec"];
            let old_job = PathBuf::from(previous["job_dir"].as_str().ok_or("job_dirが不正です")?);
            let old_output = PathBuf::from(previous["output"].as_str().ok_or("outputが不正です")?);
            if previous["context_seconds"] != context
                || previous["roi"] != roi
                || previous["processing_options"] != options
                || old_job.join("manifest.json").exists()
                || old_output.exists()
            {
                session["spec"] = new_spec(
                    &PathBuf::from(previous["source"].as_str().ok_or("sourceが不正です")?),
                    &self.directory,
                    &self.backend,
                )?;
            }
            session["spec"]["context_seconds"] = context;
            session["spec"]["roi"] = roi;
            session["spec"]["processing_options"] = options;
            session["spec"]["keep_intermediates"] = json!(true);
        }
        // Retention is now the default even for sessions saved by an older host.
        session["spec"]["keep_intermediates"] = json!(true);
        let spec = session["spec"].clone();
        let job = PathBuf::from(spec["job_dir"].as_str().ok_or("job_dirが不正です")?);
        if crate::history::owner_busy(&job) {
            return Err("このjobは別workerが実行中です。保存が終わるまで待ってください".into());
        }
        atomic_json(&sidecar(&job, "host-spec.json"), &spec)?;
        atomic_json(&self.directory.join("session.json"), &session)?;
        atomic_json(&self.directory.join("jobs/recent.json"), &spec)?;
        let _ = fs::remove_file(sidecar(&job, "host-cancel.json"));
        {
            let mut state = self.state.lock().unwrap();
            state.session = session;
            state.busy = true;
            state.value = json!({"state":"running","phase":"入力の時刻・音声を確認しています","started_at":now()});
        }
        let snapshot = self.snapshot();
        if let Err(error) = atomic_json(&self.directory.join("host-state.json"), &snapshot)
            .and_then(|_| {
                crate::history::remember(&self.directory, &snapshot["session"], &snapshot)
            })
        {
            self.state.lock().unwrap().busy = false;
            return Err(error);
        }
        let host = self.clone();
        thread::spawn(move || {
            let mut last_saved = now();
            let result = host.runtime.execute(
                &spec,
                |event| {
                    {
                        let mut state = host.state.lock().unwrap();
                        if event.get("phase").is_some() && event["phase"] != state.value["phase"] {
                            for key in [
                                "frame",
                                "total",
                                "elapsed_seconds",
                                "node_stage",
                                "prepass_frame",
                                "prepass_total",
                            ] {
                                state.value.as_object_mut().unwrap().remove(key);
                            }
                        }
                        for (key, value) in event.as_object().unwrap() {
                            state.value[key] = value.clone();
                        }
                        state.value["updated_at"] = json!(now());
                    }
                    if now() - last_saved >= 0.25 {
                        let snapshot = host.snapshot();
                        let _ = atomic_json(&host.directory.join("host-state.json"), &snapshot);
                        let _ = crate::history::remember(
                            &host.directory,
                            &snapshot["session"],
                            &snapshot,
                        );
                        last_saved = now();
                    }
                },
                Some(host.stopping.clone()),
            );
            // Intermediates remain until the user explicitly cleans a history job.
            {
                let mut state = host.state.lock().unwrap();
                let elapsed = now() - state.value["started_at"].as_f64().unwrap_or(now());
                match result {
                    Ok(result) => {
                        state.value["state"] = result["state"].clone();
                        state.value["phase"] = json!(if result["state"] == "complete" {
                            "処理完了。補正結果の見た目をご確認ください"
                        } else {
                            "停止しました。再開できます"
                        });
                        state.value["result"] = result;
                    }
                    Err(error) => {
                        state.value["state"] = json!("error");
                        state.value["phase"] = json!(error);
                    }
                }
                state.value["finished_at"] = json!(now());
                state.value["wall_seconds"] = json!(elapsed);
                state.busy = false;
            }
            let snapshot = host.snapshot();
            let _ = atomic_json(&host.directory.join("host-state.json"), &snapshot);
            let _ = crate::history::remember(&host.directory, &snapshot["session"], &snapshot);
        });
        Ok(self.snapshot())
    }
    pub fn cancel(&self) -> Result<Value, String> {
        let mut state = self.state.lock().unwrap();
        if state.busy {
            let job = PathBuf::from(
                state.session["spec"]["job_dir"]
                    .as_str()
                    .ok_or("job_dirが不正です")?,
            );
            atomic_json(&sidecar(&job, "host-cancel.json"), &json!({"cancel":true}))?;
            state.value["phase"] =
                json!("取消し要求を受け付けました。現在の窓を保存して停止します");
        }
        drop(state);
        Ok(self.snapshot())
    }
}
pub fn validate_roi(roi: &Value, width: i64, height: i64) -> Result<(), String> {
    if roi.is_null() {
        return Ok(());
    }
    let row = roi.as_object().ok_or("ROIは矩形objectです")?;
    if row.len() != 4 {
        return Err("ROIは整数のx,y,width,heightです".into());
    }
    let get = |key: &str| row.get(key).and_then(Value::as_i64).ok_or("ROIは整数です");
    let (x, y, w, h) = (get("x")?, get("y")?, get("width")?, get("height")?);
    if x < 0
        || y < 0
        || w <= 0
        || h <= 0
        || x.checked_add(w).is_none_or(|n| n > width)
        || y.checked_add(h).is_none_or(|n| n > height)
    {
        return Err("ROIが原寸の外にあります".into());
    }
    Ok(())
}
pub fn unique_output(source: &Path, job_id: &str) -> Result<PathBuf, String> {
    let suffix = if source
        .extension()
        .is_some_and(|s| s.eq_ignore_ascii_case("nut"))
    {
        "nut"
    } else {
        "mp4"
    };
    let stem = source
        .file_stem()
        .ok_or("入力名が不正です")?
        .to_string_lossy();
    let mut output = source.with_file_name(format!("{stem}.refined-{job_id}.{suffix}"));
    let mut index = 1;
    while output.exists() {
        output = source.with_file_name(format!("{stem}.refined-{job_id}-{index}.{suffix}"));
        index += 1;
    }
    Ok(output)
}
pub fn new_spec(source: &Path, directory: &Path, backend: &str) -> Result<Value, String> {
    let job_id = identifier();
    let output = unique_output(source, &job_id)?;
    Ok(
        json!({"source":source,"output":output,"job_dir":directory.join("jobs").join(job_id),"backend":backend,"roi":null,"keep_intermediates":true,
        "context_seconds":"auto","processing_options":{"mode":"balanced","render_scale_percent":100,"max_pixels":262144,
        "crop_margin":32,"sampling":"standard","version":3,"lora_profile":"unlocked_v2","lora_strength":1.0}}),
    )
}
pub fn completed(session: &Value) -> Result<Option<Value>, String> {
    if session.is_null() {
        return Ok(None);
    }
    let spec = &session["spec"];
    let Some(job) = spec["job_dir"].as_str() else {
        return Ok(None);
    };
    let job = PathBuf::from(job);
    let result = (|| -> Result<Value, String> {
        let data = saved_record(&job)?;
        let source = PathBuf::from(spec["source"].as_str().ok_or("sourceが不正です")?);
        let output = PathBuf::from(spec["output"].as_str().ok_or("outputが不正です")?);
        if data.state != "complete"
            || data.source.as_deref() != absolute(&source)?.to_str()
            || data.output.as_deref() != absolute(&output)?.to_str()
            || ![
                "pts_exact",
                "tail_duration_exact",
                "audio_packets_exact",
                "full_decode",
            ]
            .iter()
            .all(|k| data.verification[k] == true)
            || data.output_sha256.as_deref() != Some(hash(&output)?.as_str())
            || data.config.source_sha256.as_deref() != Some(hash(&source)?.as_str())
        {
            return Err("完成条件不一致".into());
        }
        let mut result = data.summary();
        result["output_integrity_checked"] = json!(true);
        Ok(result)
    })();
    Ok(result.ok())
}
