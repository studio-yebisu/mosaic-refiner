//! Small durable session records. No preview or inference is started by history.
use crate::manager::{absolute, atomic_json, now, read_json, sidecar, summary, Host};
use fs2::FileExt;
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

fn linked(path: &Path) -> bool {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return false;
    };
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if meta.file_attributes() & 0x400 != 0 {
            return true;
        }
    }
    meta.file_type().is_symlink()
}

pub fn managed_job(root: &Path, id: &str) -> Result<PathBuf, String> {
    if id.is_empty()
        || id.len() > 100
        || !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
    {
        return Err("履歴のjob IDが不正です".into());
    }
    let jobs = root.join("jobs");
    let job = jobs.join(id);
    if linked(&jobs) || linked(&job) {
        return Err("リンク先のjobは履歴から操作できません".into());
    }
    for suffix in ["host-spec.json", "host-history.json", "host-owner.lock"] {
        if linked(&sidecar(&job, suffix)) {
            return Err("リンク先の履歴は操作できません".into());
        }
    }
    Ok(job)
}

pub fn owner_busy(job: &Path) -> bool {
    let path = sidecar(&job, "host-owner.lock");
    if !path.exists() {
        return false;
    }
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_or(true, |file| file.try_lock_exclusive().is_err())
}

pub fn remember(root: &Path, session: &Value, state: &Value) -> Result<(), String> {
    let path = PathBuf::from(
        session["spec"]["job_dir"]
            .as_str()
            .ok_or("job_dirが不正です")?,
    );
    let id = path
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or("job IDが不正です")?;
    let job = managed_job(root, id)?;
    if absolute(&path)? != absolute(&job)? {
        return Err("履歴の保存先がこのアプリのjobs外です".into());
    }
    let mut session = session.clone();
    session["preview"] = Value::Null;
    session["preview_enabled"] = json!(false);
    let mut progress = json!({});
    for key in [
        "state",
        "phase",
        "frame",
        "total",
        "window",
        "window_total",
        "prepass_frame",
        "prepass_total",
        "started_at",
        "finished_at",
        "wall_seconds",
        "updated_at",
    ] {
        if let Some(value) = state.get(key) {
            progress[key] = value.clone();
        }
    }
    atomic_json(
        &sidecar(&job, "host-history.json"),
        &json!({"version":1,"session":session,"progress":progress,"saved_at":now()}),
    )
}

fn spec(root: &Path, id: &str) -> Result<Value, String> {
    let job = managed_job(root, id)?;
    let value = read_json(&sidecar(&job, "host-spec.json"))?;
    if value["job_dir"]
        .as_str()
        .map(PathBuf::from)
        .map(|p| absolute(&p))
        .transpose()?
        != Some(absolute(&job)?)
    {
        return Err("履歴のjob保存先が一致しません".into());
    }
    Ok(value)
}

pub fn session(host: &Host, id: &str) -> Result<Value, String> {
    let job = managed_job(&host.directory, id)?;
    if owner_busy(&job) {
        return Err("このjobは別workerが実行中です。停止・保存が終わってから開いてください".into());
    }
    let spec = spec(&host.directory, id)?;
    let mut session = if sidecar(&job, "host-history.json").is_file() {
        read_json(&sidecar(&job, "host-history.json"))?["session"].clone()
    } else {
        Value::Null
    };
    if session.is_null() {
        let source = PathBuf::from(spec["source"].as_str().ok_or("sourceが不正です")?);
        let meta = host.runtime.rpc("probe", json!({"source":source}))?;
        session = json!({"name":source.file_name().unwrap_or_default().to_string_lossy(),"width":meta["width"],
                         "height":meta["height"],"duration":meta["duration"],"copied_input":false});
    }
    session["spec"] = spec;
    session["id"] = json!(crate::manager::identifier());
    session["preview"] = Value::Null;
    session["preview_enabled"] = json!(false);
    Ok(session)
}

pub fn list(host: &Host) -> Result<Value, String> {
    let root = &host.directory;
    let jobs = root.join("jobs");
    if !jobs.exists() {
        return Ok(json!({"jobs":[],"warnings":[]}));
    }
    if linked(&jobs) {
        return Err("リンク先のjobsは履歴から操作できません".into());
    }
    let mut ids = BTreeSet::new();
    for entry in fs::read_dir(&jobs).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(id) = name.strip_suffix(".host-spec.json") {
            ids.insert(id.to_owned());
        } else if entry.file_type().is_ok_and(|kind| kind.is_dir())
            && entry.path().join("manifest.json").is_file()
        {
            ids.insert(name);
        }
    }
    let active = host.snapshot();
    let mut records = Vec::new();
    let mut warnings = Vec::new();
    for id in ids {
        let result = (|| -> Result<Value, String> {
            let job = managed_job(root, &id)?;
            let manifest = job.join("manifest.json");
            if linked(&manifest) {
                return Err("リンク先の処理記録です".into());
            }
            let saved = if manifest.is_file() {
                summary(&job)?
            } else {
                json!({})
            };
            let history = sidecar(&job, "host-history.json");
            let entry = if history.is_file() {
                read_json(&history)?
            } else {
                json!({})
            };
            let selected = active["session"]["spec"]["job_dir"]
                .as_str()
                .map(PathBuf::from)
                == Some(job.clone());
            let spec = if sidecar(&job, "host-spec.json").is_file() {
                spec(root, &id)?
            } else {
                json!({})
            };
            let progress = if selected && (active["busy"] == true || active["state"] == "error") {
                active.clone()
            } else {
                entry["progress"].clone()
            };
            let mut state = progress["state"]
                .as_str()
                .or(saved["state"].as_str())
                .unwrap_or("ready")
                .to_owned();
            if saved["state"] == "complete" {
                state = "complete".into();
            }
            let owned = owner_busy(&job);
            if owned {
                state = "running".into();
            }
            if state == "running" && !owned && !(selected && active["busy"] == true) {
                state = "interrupted".into();
            }
            let source = PathBuf::from(spec["source"].as_str().unwrap_or(""));
            let output = PathBuf::from(spec["output"].as_str().unwrap_or(""));
            let cleanup = if job.join("cleanup.json").is_file()
                && !linked(&job.join("cleanup.json"))
            {
                let report = read_json(&job.join("cleanup.json"))?;
                json!({"state":report["state"],"bytes_freed":report["bytes_freed"],"skipped_count":report["skipped"].as_array().map_or(0,Vec::len)})
            } else {
                Value::Null
            };
            let updated = entry["saved_at"].as_f64().unwrap_or_else(|| {
                fs::metadata(&manifest)
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0.0, |d| d.as_secs_f64())
            });
            let selectable =
                !owned && !spec.is_null() && spec.get("source").is_some() && source.is_file();
            Ok(
                json!({"id":id,"name":entry["session"]["name"].as_str().map(str::to_owned)
                .unwrap_or_else(||source.file_name().unwrap_or_default().to_string_lossy().into_owned()),
                "state":state,"phase":progress["phase"],"frame":progress["frame"],"total":progress["total"],
                "completed_units":saved["completed_units"],"total_units":saved["total_units"],"updated_at":updated,
                "selected":selected,"selectable":selectable,"reason":if selectable {""} else if owned {"別workerが実行中です。停止・保存が終わるまで待ってください"} else {"元動画またはこの版の再開条件がありません"},
                "output_exists":output.is_file(),"cleanup_allowed":!owned && state=="complete" && spec.get("job_dir").is_some() && output.is_file(),
                "cleanup":cleanup}),
            )
        })();
        match result {
            Ok(value) => records.push(value),
            Err(reason) => warnings.push(json!({"id":id,"reason":reason})),
        }
    }
    records.sort_by(|a, b| {
        b["updated_at"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["updated_at"].as_f64().unwrap_or(0.0))
    });
    Ok(json!({"jobs":records,"warnings":warnings}))
}

pub fn cleanup(host: &Host, id: &str) -> Result<Value, String> {
    let job = managed_job(&host.directory, id)?;
    if owner_busy(&job) {
        return Err("このjobは別workerが実行中です".into());
    }
    spec(&host.directory, id)?;
    if linked(&job.join("manifest.json")) || linked(&job.join("cleanup.json")) {
        return Err("リンク先の処理記録は整理できません".into());
    }
    host.runtime
        .rpc("cleanup", json!({"job_dir":job,"manual":true}))
}
