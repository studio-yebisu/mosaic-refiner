//! Native application owner. Python is a one-shot media/inference worker,
//! never an HTTP server or job controller on this path.
mod history;
mod manager;
mod server;
use clap::{Arg, ArgAction, Command};
use manager::{atomic_json, read_json, summary, Runtime};
use serde_json::{json, Value};
use std::{env, fs, path::PathBuf, process};

fn user_web_directory() -> Result<PathBuf, String> {
    let root = env::var_os("LOCALAPPDATA").map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .ok_or("LOCALAPPDATAが未設定または相対パスです。web --directoryで利用者専用の保存先を指定してください")?;
    Ok(root.join("MosaicRefiner/web"))
}

fn cli() -> Command {
    let schema: Value = serde_json::from_str(include_str!("cli-schema.json")).unwrap();
    let mut root = Command::new("mosaic-refiner")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Rust動画補正アプリ。AI推論は選択済みの環境を使います")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(Arg::new("project-root").long("project-root").global(true))
        .arg(Arg::new("worker-python").long("worker-python").global(true));
    for row in schema.as_array().unwrap() {
        let name = row["name"].as_str().unwrap();
        let mut sub = Command::new(name.to_owned());
        for item in row["arguments"].as_array().unwrap() {
            let mut arg = Arg::new(item["name"].as_str().unwrap().to_owned())
                .help(item["help"].as_str().unwrap().to_owned())
                .required(item["required"].as_bool().unwrap());
            for flag in item["flags"].as_array().unwrap() {
                let flag = flag.as_str().unwrap();
                if let Some(long) = flag.strip_prefix("--") {
                    arg = arg.long(long.to_owned());
                }
            }
            if item["boolean"].as_bool().unwrap() {
                arg = arg.action(ArgAction::SetTrue);
            }
            if item["append"].as_bool().unwrap() {
                arg = arg.action(ArgAction::Append);
            }
            if let Some(values) = item["choices"].as_array() {
                arg = arg.value_parser(clap::builder::PossibleValuesParser::new(
                    values
                        .iter()
                        .map(|v| {
                            if let Some(s) = v.as_str() {
                                s.to_owned()
                            } else {
                                v.to_string()
                            }
                        })
                        .collect::<Vec<_>>(),
                ));
            }
            sub = sub.arg(arg);
        }
        if name == "web" {
            sub = sub.arg(Arg::new("directory").long("directory")).arg(
                Arg::new("backend")
                    .long("backend")
                    .value_parser(["h3", "identity", "invert"])
                    .default_value("h3"),
            );
        }
        root = root.subcommand(sub);
    }
    root.subcommand(Command::new("_worker").hide(true))
}

fn run() -> Result<i32, String> {
    let matches = cli().get_matches();
    let (name, args) = matches.subcommand().unwrap();
    let executable = env::current_exe().map_err(|e| e.to_string())?;
    let root = matches
        .get_one::<String>("project-root")
        .map(PathBuf::from)
        .or_else(|| env::var_os("MOSAIC_REFINER_PROJECT_ROOT").map(PathBuf::from))
        .unwrap_or_else(|| {
            executable
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        });
    let python = matches
        .get_one::<String>("worker-python")
        .map(PathBuf::from)
        .or_else(|| env::var_os("MOSAIC_REFINER_WORKER_PYTHON").map(PathBuf::from))
        .unwrap_or_else(|| {
            root.join(if cfg!(windows) {
                ".venv/Scripts/python.exe"
            } else {
                ".venv/bin/python"
            })
        });
    let runtime = Runtime { root, python };
    runtime.validate()?;
    if name == "_worker" {
        let request: Value =
            serde_json::from_reader(std::io::stdin()).map_err(|e| e.to_string())?;
        let job = PathBuf::from(
            request["data"]["job_dir"]
                .as_str()
                .ok_or("job_dirがありません")?,
        );
        if let Some(parent) = job.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        manager::migrate_legacy_spec(&job)?;
        let owner = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(manager::sidecar(&job, "host-owner.lock"))
            .map_err(|e| e.to_string())?;
        fs2::FileExt::try_lock_exclusive(&owner).map_err(|_| "このjobは別workerが実行中です")?;
        let result = runtime.python_worker(
            request["operation"]
                .as_str()
                .ok_or("operationがありません")?,
            request["data"].clone(),
            |event| println!("{}", json!({"kind":"progress","data":event})),
        );
        return match result {
            Ok(result) => {
                println!("{}", json!({"kind":"result","data":result}));
                Ok(0)
            }
            Err(error) => {
                println!("{}", json!({"kind":"error","data":{"error":error}}));
                Ok(2)
            }
        };
    }
    if name == "status" || name == "cancel" {
        let directory = PathBuf::from(args.get_one::<String>("job_dir").unwrap());
        manager::migrate_legacy_spec(&directory)?;
        let result = if name == "status" {
            summary(&directory)?
        } else {
            if !directory.join("manifest.json").is_file()
                && !manager::sidecar(&directory, "host-spec.json").is_file()
            {
                return Err("jobのmanifest/所有記録がありません".into());
            }
            atomic_json(
                &manager::sidecar(&directory, "host-cancel.json"),
                &json!({"cancel":true}),
            )?;
            if directory.join("manifest.json").is_file() {
                atomic_json(&directory.join("cancel.json"), &json!({"cancel":true}))?;
            }
            json!({"cancel_requested":true})
        };
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return Ok(0);
    }
    if name == "gui" {
        let directory = user_web_directory()?;
        return server::serve(runtime, directory, 8765, "h3".into(), true);
    }
    if name == "web" {
        let port = args
            .get_one::<String>("port")
            .map(|s| s.parse::<u16>())
            .transpose()
            .map_err(|_| "portが無効です")?
            .unwrap_or(8765);
        let directory = args
            .get_one::<String>("directory")
            .map(PathBuf::from)
            .map(Ok)
            .unwrap_or_else(user_web_directory)?;
        return server::serve(
            runtime,
            directory,
            port,
            args.get_one::<String>("backend").unwrap().clone(),
            false,
        );
    }
    if name == "refine" {
        let get = |s: &str| args.get_one::<String>(s).cloned();
        let integer = |s: &str, default: i64| -> Result<i64, String> {
            get(s)
                .map(|v| v.parse::<i64>())
                .transpose()
                .map_err(|_| format!("{s}は整数です"))
                .map(|v| v.unwrap_or(default))
        };
        let mut options = json!({"mode":get("mode").unwrap_or("balanced".into()),
            "lora_profile":get("h3_lora").unwrap_or("unlocked_v2".into()),
            "lora_strength":get("lora_strength").unwrap_or("1".into()).parse::<f64>().map_err(|_|"LoRA強度が無効です")?});
        for (arg, key) in [
            ("render_scale", "render_scale_percent"),
            ("pixel_budget", "max_pixels"),
            ("crop_margin", "crop_margin"),
        ] {
            if get(arg).is_some() {
                options[key] = json!(integer(arg, 0)?);
            }
        }
        if let Some(s) = get("sampling") {
            options["sampling"] = json!(s);
        }
        options = runtime.rpc("options", json!({"processing_options":options}))?;
        let roi = get("roi")
            .map(|text| manager::parse_roi(&text))
            .transpose()?;
        let directory = manager::absolute(&PathBuf::from(get("job_dir").unwrap()))?;
        manager::migrate_legacy_spec(&directory)?;
        fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let source = manager::absolute(&PathBuf::from(get("input").unwrap()))?;
        let output = manager::absolute(&PathBuf::from(get("output").unwrap()))?;
        let spec = json!({"source":source,"output":output,"job_dir":directory,"backend":"h3","roi":roi,
            "context_seconds":integer("context_seconds",5)?,"unit_seconds":get("unit_seconds").unwrap_or("15".into()),
            "processing_options":options});
        if manager::sidecar(&directory, "host-spec.json").exists()
            && read_json(&manager::sidecar(&directory, "host-spec.json"))? != spec
        {
            return Err("保存済み条件と異なります。新しいjob-dirを指定してください".into());
        }
        // Ownership stays beside the job, leaving the core's unowned-directory
        // and no-overwrite contract intact before manifest creation.
        atomic_json(&manager::sidecar(&directory, "host-spec.json"), &spec)?;
        let _ = fs::remove_file(manager::sidecar(&directory, "host-cancel.json"));
        let stop = manager::install_stop_handler()?;
        let result = runtime.execute(&spec, |event| eprintln!("{}", event), Some(stop))?;
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
        return Ok(if result["state"] == "cancelled" {
            130
        } else {
            0
        });
    }
    // Compatibility operations (CPU processing/setup/planning) run once. Their
    // public arguments were already checked by clap, without a Python daemon.
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut forwarded = Vec::new();
    let mut index = 0;
    while index < raw.len() {
        if raw[index] == "--project-root" || raw[index] == "--worker-python" {
            index += 2;
        } else if raw[index].starts_with("--project-root=")
            || raw[index].starts_with("--worker-python=")
        {
            index += 1;
        } else {
            forwarded.push(raw[index].clone());
            index += 1;
        }
    }
    if name == "auth" {
        let status = process::Command::new(&runtime.python)
            .args(["-B", "-m", "mosaic_refiner"])
            .args(&forwarded)
            .current_dir(&runtime.root)
            .env("PYTHONUTF8", "1")
            .env("MOSAIC_REFINER_PYTHON_WORKER", "1")
            .status()
            .map_err(|_| "本人の認証端末を起動できません")?;
        return Ok(status.code().unwrap_or(2));
    }
    let result = if name == "run" {
        let job = manager::absolute(&PathBuf::from(args.get_one::<String>("job_dir").unwrap()))?;
        manager::migrate_legacy_spec(&job)?;
        let _ = fs::remove_file(manager::sidecar(&job, "host-cancel.json"));
        runtime.supervised_cli(
            json!({"argv":forwarded,"job_dir":job}),
            manager::install_stop_handler()?,
        )?
    } else {
        runtime.rpc("cli", json!({"argv":forwarded}))?
    };
    if let Some(text) = result["result"]["text"].as_str() {
        print!("{text}");
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&result["result"]).unwrap()
        );
    }
    Ok(result["exit_code"].as_i64().unwrap_or(0) as i32)
}

fn main() {
    match run() {
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!("エラー: {error}");
            process::exit(2);
        }
    }
}
