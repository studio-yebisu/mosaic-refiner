//! Independent RGB composition and numeric verification. No model or media I/O.
//! ABI callers own buffers for the entire call; writable output must not alias
//! any input. The Python adapter validates dtype, contiguity and buffer lengths.
#![deny(unsafe_op_in_unsafe_fn)]

use std::{mem, slice};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    fn valid(self, width: usize, height: usize) -> bool {
        self.width > 0 && self.height > 0
            && self.x.checked_add(self.width).is_some_and(|x| x <= width)
            && self.y.checked_add(self.height).is_some_and(|y| y <= height)
    }
}

#[repr(C)]
pub struct Window {
    pub pixels: *const u8,
    pub bytes: usize,
    pub width: usize,
    pub height: usize,
    pub content_width: usize,
    pub content_height: usize,
    pub crop_x: i64,
    pub crop_y: i64,
    pub crop_width: usize,
    pub crop_height: usize,
    pub weight: f64,
}

#[repr(C)]
#[derive(Default, Debug, PartialEq)]
pub struct Statistics {
    pub changed_channels: u64,
    pub outside_changed_channels: u64,
    pub masked_channels: u64,
    pub masked_abs_error_sum: u64,
    pub masked_channels_over8: u64,
    pub masked_channels_over24: u64,
    pub first_outside_channel: i64,
}

const DIMENSIONS: i32 = 1;
const LENGTH: i32 = 2;
const POINTER: i32 = 3;
const WEIGHT: i32 = 4;
const MASK: i32 = 5;

fn rgb_bytes(width: usize, height: usize) -> Option<usize> {
    if width == 0 || height == 0 { return None; }
    width.checked_mul(height)?.checked_mul(3).filter(|n| *n <= isize::MAX as usize)
}

fn aligned<T>(ptr: *const T) -> bool {
    !ptr.is_null() && (ptr as usize) % mem::align_of::<T>() == 0
}

#[no_mangle]
pub extern "C" fn mr_abi_version() -> u32 { 1 }

/// All pointers must reference valid non-overlapping buffers of the stated size.
#[no_mangle]
pub unsafe extern "C" fn mr_composite_rect(
    source: *const u8, source_len: usize, width: usize, height: usize,
    roi: Rect, restored: *const u8, restored_len: usize, feather: usize,
    output: *mut u8, output_len: usize,
) -> i32 {
    if !roi.valid(width, height) || feather > roi.width.min(roi.height) / 2 { return DIMENSIONS; }
    if rgb_bytes(width, height) != Some(source_len) || output_len != source_len
        || rgb_bytes(roi.width, roi.height) != Some(restored_len) { return LENGTH; }
    if !aligned(source) || !aligned(restored) || !aligned(output) { return POINTER; }
    let source = unsafe { slice::from_raw_parts(source, source_len) };
    let restored = unsafe { slice::from_raw_parts(restored, restored_len) };
    let output = unsafe { slice::from_raw_parts_mut(output, output_len) };
    output.copy_from_slice(source);
    for y in 0..roi.height {
        for x in 0..roi.width {
            let edge = x.min(y).min(roi.width - 1 - x).min(roi.height - 1 - y);
            let alpha = if feather == 0 { 1.0 } else { (edge as f64 / feather as f64).min(1.0) };
            let pos = ((roi.y + y) * width + roi.x + x) * 3;
            let crop = (y * roi.width + x) * 3;
            for c in 0..3 {
                let old = source[pos + c] as f64;
                output[pos + c] = (old + alpha * (restored[crop + c] as f64 - old)).round_ties_even() as u8;
            }
        }
    }
    0
}

/// Fused inverse crop, ordered temporal blending and inward-mask composition.
/// Rounding and operation order match the existing NumPy float64 compositor.
#[no_mangle]
pub unsafe extern "C" fn mr_composite_windows(
    source: *const u8, source_len: usize, width: usize, height: usize, roi: Rect,
    alpha: *const f32, alpha_len: usize, windows: *const Window, window_count: usize,
    output: *mut u8, output_len: usize,
) -> i32 {
    if !roi.valid(width, height) { return DIMENSIONS; }
    if rgb_bytes(width, height) != Some(source_len) || output_len != source_len
        || roi.width.checked_mul(roi.height) != Some(alpha_len) { return LENGTH; }
    if window_count == 0 || window_count > 64 { return WEIGHT; }
    if !aligned(source) || !aligned(output) || !aligned(alpha) || !aligned(windows) { return POINTER; }
    let windows = unsafe { slice::from_raw_parts(windows, window_count) };
    let alpha = unsafe { slice::from_raw_parts(alpha, alpha_len) };
    if alpha.iter().any(|a| !a.is_finite() || !(0.0..=1.0).contains(a)) { return MASK; }
    let mut total = 0.0;
    for w in windows {
        if w.content_width == 0 || w.content_height == 0 || w.content_width > w.width
            || w.content_height > w.height || w.crop_width == 0 || w.crop_height == 0 { return DIMENSIONS; }
        if rgb_bytes(w.width, w.height) != Some(w.bytes) { return LENGTH; }
        if !aligned(w.pixels) { return POINTER; }
        if !w.weight.is_finite() || w.weight <= 0.0 { return WEIGHT; }
        total += w.weight;
    }
    if !total.is_finite() { return WEIGHT; }
    let maps: Vec<_> = windows.iter().map(|w| {
        let xs: Vec<_> = (0..roi.width).map(|x| {
            (((roi.x + x) as f64 - w.crop_x as f64 + 0.5) * w.content_width as f64
                / w.crop_width as f64).trunc().clamp(0.0, (w.content_width - 1) as f64) as usize
        }).collect();
        let ys: Vec<_> = (0..roi.height).map(|y| {
            (((roi.y + y) as f64 - w.crop_y as f64 + 0.5) * w.content_height as f64
                / w.crop_height as f64).trunc().clamp(0.0, (w.content_height - 1) as f64) as usize
        }).collect();
        (xs, ys)
    }).collect();
    let pixels: Vec<_> = windows.iter().map(|w| unsafe { slice::from_raw_parts(w.pixels, w.bytes) }).collect();
    let source = unsafe { slice::from_raw_parts(source, source_len) };
    let output = unsafe { slice::from_raw_parts_mut(output, output_len) };
    output.copy_from_slice(source);
    for y in 0..roi.height {
        for x in 0..roi.width {
            let a = alpha[y * roi.width + x] as f64;
            if a == 0.0 { continue; }
            let mut restored = [0.0_f64; 3];
            for (i, w) in windows.iter().enumerate() {
                let (xs, ys) = &maps[i];
                let offset = (ys[y] * w.width + xs[x]) * 3;
                for c in 0..3 { restored[c] += pixels[i][offset + c] as f64 * w.weight; }
            }
            let pos = ((roi.y + y) * width + roi.x + x) * 3;
            for c in 0..3 {
                let old = source[pos + c] as f64;
                let value = old + a * (restored[c] / total - old);
                output[pos + c] = value.round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }
    0
}

/// A zero-sized ROI with alpha_len=0 means that every pixel is outside.
/// Source and result are read-only; they may be the same buffer.
#[no_mangle]
pub unsafe extern "C" fn mr_compare_rgb(
    source: *const u8, result: *const u8, bytes: usize, width: usize, height: usize,
    roi: Rect, alpha: *const f32, alpha_len: usize, stats: *mut Statistics,
) -> i32 {
    if rgb_bytes(width, height) != Some(bytes) { return LENGTH; }
    let empty = roi.width == 0 && roi.height == 0 && alpha_len == 0;
    if !empty && !roi.valid(width, height) { return DIMENSIONS; }
    if !empty && roi.width.checked_mul(roi.height) != Some(alpha_len) { return LENGTH; }
    if !aligned(source) || !aligned(result) || !aligned(stats) || (!empty && !aligned(alpha)) { return POINTER; }
    let alpha = if empty { &[][..] } else { unsafe { slice::from_raw_parts(alpha, alpha_len) } };
    if alpha.iter().any(|a| !a.is_finite() || !(0.0..=1.0).contains(a)) { return MASK; }
    let source = unsafe { slice::from_raw_parts(source, bytes) };
    let result = unsafe { slice::from_raw_parts(result, bytes) };
    let mut s = Statistics { first_outside_channel: -1, ..Statistics::default() };
    for y in 0..height {
        for x in 0..width {
            let active = !empty && x >= roi.x && x < roi.x + roi.width
                && y >= roi.y && y < roi.y + roi.height
                && alpha[(y - roi.y) * roi.width + x - roi.x] > 0.0;
            let pos = (y * width + x) * 3;
            for c in 0..3 {
                let d = source[pos + c].abs_diff(result[pos + c]) as u64;
                s.changed_channels += u64::from(d != 0);
                if active {
                    s.masked_channels += 1;
                    s.masked_abs_error_sum += d;
                    s.masked_channels_over8 += u64::from(d > 8);
                    s.masked_channels_over24 += u64::from(d > 24);
                } else if d != 0 {
                    s.outside_changed_channels += 1;
                    if s.first_outside_channel < 0 { s.first_outside_channel = (pos + c) as i64; }
                }
            }
        }
    }
    unsafe { stats.write(s); }
    0
}
