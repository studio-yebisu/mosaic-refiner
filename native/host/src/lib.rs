//! Small C ABI for durable job records. No media/model operations are exported.
#![allow(dead_code)]
mod history;
mod manager;
use fs2::FileExt;
use serde_json::{json, Value};
use std::{
    ffi::c_void,
    fs::{File, OpenOptions},
    path::PathBuf,
    ptr, slice,
};

#[no_mangle]
pub extern "C" fn mr_host_abi_version() -> u32 {
    1
}

/// Borrowed UTF-8 JSON request; output is owned until mr_host_free. Callers
/// provide valid nonaliased buffers and writable output slots.
#[no_mangle]
pub unsafe extern "C" fn mr_host_json(
    input: *const u8,
    length: usize,
    output: *mut *mut u8,
    size: *mut usize,
) -> i32 {
    if input.is_null() || output.is_null() || size.is_null() {
        return 1;
    }
    let result = std::panic::catch_unwind(|| -> Result<Value, String> {
        let request: Value = serde_json::from_slice(slice::from_raw_parts(input, length))
            .map_err(|e| e.to_string())?;
        let path = PathBuf::from(request["path"].as_str().ok_or("pathが必要です")?);
        match request["operation"].as_str() {
            Some("atomic") => {
                manager::atomic_json(&path, &request["data"])?;
                Ok(json!({"saved":true}))
            }
            Some("load") => {
                let value = manager::read_json(&path)?;
                if !value.is_object() {
                    return Err("JSON objectが必要です".into());
                }
                Ok(value)
            }
            Some("hash") => Ok(json!(manager::hash(&path)?)),
            Some("summary") => manager::summary(&path),
            _ => Err("未知のjob operation".into()),
        }
    })
    .unwrap_or_else(|_| Err("job core panic".into()));
    let envelope = match result {
        Ok(value) => json!({"ok":true,"data":value}),
        Err(error) => json!({"ok":false,"error":error}),
    };
    let bytes = envelope.to_string().into_bytes().into_boxed_slice();
    *size = bytes.len();
    *output = Box::into_raw(bytes) as *mut u8;
    0
}
#[no_mangle]
pub unsafe extern "C" fn mr_host_free(pointer: *mut u8, length: usize) {
    if !pointer.is_null() {
        drop(Box::from_raw(ptr::slice_from_raw_parts_mut(
            pointer, length,
        )));
    }
}
#[no_mangle]
pub unsafe extern "C" fn mr_host_lock(input: *const u8, length: usize) -> *mut c_void {
    if input.is_null() {
        return ptr::null_mut();
    }
    let result = (|| -> Result<File, String> {
        let path = PathBuf::from(
            std::str::from_utf8(slice::from_raw_parts(input, length)).map_err(|e| e.to_string())?,
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|e| e.to_string())?;
        file.try_lock_exclusive().map_err(|e| e.to_string())?;
        Ok(file)
    })();
    match result {
        Ok(file) => Box::into_raw(Box::new(file)) as *mut c_void,
        Err(_) => ptr::null_mut(),
    }
}
#[no_mangle]
pub unsafe extern "C" fn mr_host_unlock(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut File));
    }
}
