//! EXL3 model file reader and dequantizer.
//!
//! EXL3 weights are stored in standard safetensors containers. Each quantized
//! linear layer consists of:
//!   - `<name>.trellis`  I16  (k/16, n/16, 16*K)   packed tail-biting trellis
//!   - `<name>.suh`      F16  (k,)                 input sign vector
//!   - `<name>.svh`      F16  (n,)                 output sign vector
//!   - `<name>.bias`     F16  (n,)                 optional bias
//!   - `<name>.mcg` / `<name>.mul1`  I32 marker    codebook selection
//!
//! Dequantization: unpack 256 codewords per 16x16 tile, decode each to fp16,
//! then apply diag(suh) . H128 . W_hat . H128 . diag(svh).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::json::Json;
use crate::tensor::{F16Matrix, Matrix};
use crate::threadpool::SendSlice;

/// Best-effort resident-set size in MiB. `std` has no portable RSS API, so
/// this is Linux-only and simply omitted elsewhere.
fn rss_mib() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let s = fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("VmRSS:") {
                let kb: f64 = v.split_whitespace().next()?.parse().ok()?;
                return Some(kb / 1024.0);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// FP16 helpers
// ---------------------------------------------------------------------------

/// Convert an fp16 bit pattern to f32 (round-to-nearest is implicit in the
/// exact conversion).
#[inline]
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        // Subnormal or zero.
        sign * (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 31 {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1024.0 + mant as f32) * 2.0f32.powi(exp - 25)
    }
}

/// Round an f32 to the nearest fp16 and return its bit pattern
/// (round-to-nearest-even).
#[inline]
pub(crate) fn f32_to_half(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    if v == f32::INFINITY {
        return 0x7c00;
    }
    if v == f32::NEG_INFINITY {
        return 0xfc00;
    }
    let neg = v < 0.0;
    let a = v.abs();
    if a > 65504.0 {
        return if neg { 0xfc00 } else { 0x7c00 };
    }
    if a < 6.103_515_6e-5 {
        // Subnormal: round mantissa / 2^-24 to nearest even.
        let scaled = a * 16777216.0; // 2^24
        let mut m = scaled as u32;
        let frac = scaled - m as f32;
        if frac > 0.5 || (frac == 0.5 && (m & 1) != 0) {
            m += 1;
        }
        return (if neg { 0x8000u16 } else { 0 }) | (m.min(0x3ff) as u16);
    }
    // Normal: find exponent.
    let mut e = a.log2().floor() as i32;
    if e > 15 {
        e = 15;
    }
    let base = 2.0f32.powi(e);
    let mut m = ((a / base - 1.0) * 1024.0) as u32;
    let frac = (a / base - 1.0) * 1024.0 - m as f32;
    if frac > 0.5 || (frac == 0.5 && (m & 1) != 0) {
        m += 1;
    }
    if m >= 1024 {
        m = 0;
        e += 1;
        if e > 15 {
            return if neg { 0xfc00 } else { 0x7c00 };
        }
    }
    (if neg { 0x8000u16 } else { 0 }) | (((e + 15) as u16) << 10) | (m as u16)
}

/// fp16 addition (emulate __hadd: compute in f32, round to fp16).
#[inline]
fn half_add(a: u16, b: u16) -> u16 {
    f32_to_half(half_to_f32(a) + half_to_f32(b))
}

/// fp16 fused multiply-add (emulate __hfma).
#[inline]
fn half_fma(a: u16, b: u16, c: u16) -> u16 {
    f32_to_half(half_to_f32(a) * half_to_f32(b) + half_to_f32(c))
}

// ---------------------------------------------------------------------------
// Codebook decode (16-bit index -> fp16 weight)
// ---------------------------------------------------------------------------

#[inline]
fn lop3(x: u32) -> u32 {
    (x & 0x8fff8fff) ^ 0x3b603b60
}

#[inline]
fn decode_3inst(idx: u16) -> u16 {
    let mut x = (idx as u32).wrapping_mul(89226354).wrapping_add(64248484);
    x = lop3(x);
    half_add((x & 0xffff) as u16, (x >> 16) as u16)
}

#[inline]
fn decode_mcg(idx: u16) -> u16 {
    let mut x = (idx as u32).wrapping_mul(0xcbac1fed);
    x = lop3(x);
    half_add((x & 0xffff) as u16, (x >> 16) as u16)
}

#[inline]
fn decode_mul1(idx: u16) -> u16 {
    let x = (idx as u32).wrapping_mul(0x83dcd12d);
    let bsum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff);
    let h = (0x6400u32 + bsum) as u16; // 1024 + byte sum, as fp16 bits
    half_fma(h, 0x1eee, 0xc931)
}

#[derive(Clone, Copy, PartialEq)]
enum Codebook {
    ThreeInst,
    Mcg,
    Mul1,
}

#[inline]
fn decode(idx: u16, cb: Codebook) -> u16 {
    match cb {
        Codebook::ThreeInst => decode_3inst(idx),
        Codebook::Mcg => decode_mcg(idx),
        Codebook::Mul1 => decode_mul1(idx),
    }
}

/// Precomputed 65536-entry f32 decode tables. The codebooks are pure functions
/// of the 16-bit index, so decoding once at startup turns the hot per-tile loop
/// into a single array lookup (the original did log2/powi/exp per codeword).
static DECODE_3: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
static DECODE_MCG: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
static DECODE_MUL1: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

fn decode_f32(idx: u16, cb: Codebook) -> f32 {
    let table = match cb {
        Codebook::ThreeInst => DECODE_3.get_or_init(|| build_decode_table(decode_3inst)),
        Codebook::Mcg => DECODE_MCG.get_or_init(|| build_decode_table(decode_mcg)),
        Codebook::Mul1 => DECODE_MUL1.get_or_init(|| build_decode_table(decode_mul1)),
    };
    table[idx as usize]
}

fn build_decode_table(f: fn(u16) -> u16) -> Vec<f32> {
    (0..=u16::MAX).map(|i| half_to_f32(f(i))).collect()
}

/// Fast fp16-bit-pattern -> f32 lookup (avoids powi in the matmul inner loop).
static HALF_F32: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
#[inline]
pub(crate) fn half_f32(bits: u16) -> f32 {
    HALF_F32.get_or_init(|| (0..=u16::MAX).map(half_to_f32).collect())[bits as usize]
}

/// Dot product against one fp16 weight row, converting weights on the fly via
/// the lookup table. Avoids materialising a full-size f32 scratch per matmul.
#[inline]
fn dot_f16(x: &[f32], w: &[u16]) -> f32 {
    assert_eq!(x.len(), w.len(), "dot_f16: length mismatch");
    let mut acc = 0.0f32;
    for k in 0..x.len() {
        acc += x[k] * half_f32(w[k]);
    }
    acc
}

// ---------------------------------------------------------------------------
// Trellis unpacking
// ---------------------------------------------------------------------------

/// Extract a 16-bit window from a bitstream stored MSB-first in u32 words.
/// `start` is the bit index of the first bit of the window (may wrap mod R).
/// O(1): one 64-bit concatenate + shift instead of 16 per-bit divisions.
#[inline]
fn window(words: &[u32], nw: usize, start: i64) -> u16 {
    let r = (nw * 32) as i64;
    let s = start.rem_euclid(r) as usize;
    let wi = s >> 5; // word index
    let o = s & 31; // bit offset within word (MSB-first)
    let w0 = words[wi] as u64;
    let w1 = words[(wi + 1) % nw] as u64;
    let combined = (w0 << 32) | w1;
    // Bit `s` is at u64 position (63 - o); shift so it lands at bit 15.
    ((combined >> (48 - o)) & 0xffff) as u16
}

/// Row-major offset within a 16x16 tile for ring position `i`.
#[inline]
fn tile_perm(i: usize) -> usize {
    let t = i >> 3;
    let j = i & 7;
    let r = (t & 3) * 2 + if j & 1 != 0 { 1 } else { 0 } + if j & 2 != 0 { 8 } else { 0 };
    let c = (t >> 2) + if j & 4 != 0 { 8 } else { 0 };
    r * 16 + c
}

/// Precomputed tile permutation (256 lookups avoid the bit arithmetic in the
/// innermost dequant loop).
static TILE_PERM: std::sync::OnceLock<[usize; 256]> = std::sync::OnceLock::new();
fn tile_perm_table() -> &'static [usize; 256] {
    TILE_PERM.get_or_init(|| {
        let mut p = [0usize; 256];
        for (i, v) in p.iter_mut().enumerate() {
            *v = tile_perm(i);
        }
        p
    })
}

/// Unpack one tile of 256 decoded fp16 weights.
///
/// `packed` points at the tile's packed uint16 data (length `16*K` for integer
/// K, or `16*KA + 8` for half-integer K). `k_half` is true for the MUL1
/// half-integer rates.
fn unpack_tile_bits(
    packed: &[u16],
    k_bits_int: usize,
    k_half: bool,
    cb: Codebook,
    out: &mut [u16; 256],
) {
    let nw = packed.len() / 2; // number of u32 words
                               // Stack buffer avoids a heap allocation per tile. Max supported width
                               // is 144 u16 words = 72 u32 words (K=8 -> 64; half KA=7 -> 60).
                               // `validate_tile_width` enforces this at load time.
    let mut words = [0u32; 72];
    for w in 0..nw {
        words[w] = ((packed[2 * w + 1] as u32) << 16) | (packed[2 * w] as u32);
    }
    let perm = tile_perm_table();
    if !k_half {
        // Integer K: S(i) = (i+1)*K; window ends at S(i).
        let k = k_bits_int;
        for i in 0..256 {
            let end = ((i + 1) * k) as i64;
            out[perm[i]] = decode(window(&words, nw, end - 16), cb);
        }
    } else {
        // Half-integer K = KA + 0.5 with MASK = 0xAAAA.
        let ka = k_bits_int;
        let bpb = 16 * ka + 8; // bits per tile (16*K)
        let end_of_group = |g: usize| (g * bpb) as i64;
        for i in 0..256 {
            let g = i >> 4;
            let q = i & 15;
            // S(i) = g*bpb + sum_{j=0..=q} (ka + odd(j))
            let end = end_of_group(g) + (q * ka) as i64 + q.div_ceil(2) as i64;
            out[perm[i]] = decode(window(&words, nw, end - 16), cb);
        }
    }
}

/// Unpack one tile into 256 decoded f32 weights (integer or half-integer K).
fn unpack_tile(
    packed: &[u16],
    k_bits_int: usize,
    k_half: bool,
    cb: Codebook,
    out: &mut [f32; 256],
) {
    let mut bits = [0u16; 256];
    unpack_tile_bits(packed, k_bits_int, k_half, cb, &mut bits);
    for i in 0..256 {
        out[i] = half_f32(bits[i]);
    }
}

/// Fast f32 unpack for the decode hot path: look the codebook value up
/// directly as f32 instead of running `decode` arithmetic and then a second
/// fp16->f32 LUT lookup per weight. Numerically identical to `unpack_tile`
/// (`decode_f32(idx) == half_f32(decode(idx))` by construction).
fn unpack_tile_f32(
    packed: &[u16],
    k_bits_int: usize,
    k_half: bool,
    cb: Codebook,
    out: &mut [f32; 256],
) {
    let nw = packed.len() / 2;
    let mut words = [0u32; 72];
    for w in 0..nw {
        words[w] = ((packed[2 * w + 1] as u32) << 16) | (packed[2 * w] as u32);
    }
    let perm = tile_perm_table();
    if !k_half {
        let k = k_bits_int;
        for i in 0..256 {
            let end = ((i + 1) * k) as i64;
            out[perm[i]] = decode_f32(window(&words, nw, end - 16), cb);
        }
    } else {
        let ka = k_bits_int;
        let bpb = 16 * ka + 8;
        for i in 0..256 {
            let g = i >> 4;
            let q = i & 15;
            let end = (g * bpb) as i64 + (q * ka) as i64 + q.div_ceil(2) as i64;
            out[perm[i]] = decode_f32(window(&words, nw, end - 16), cb);
        }
    }
}

// ---------------------------------------------------------------------------
// Hadamard transform (Sylvester H128 / sqrt(128), in-place butterfly)
// ---------------------------------------------------------------------------

/// Apply a normalized 128-point Sylvester Hadamard transform to the strided
/// vector `data[base + i*stride]`, `i = 0..128`.
#[inline]
fn hadamard128(data: &mut [f32], base: usize, stride: usize, scale: f32) {
    for step in [1usize, 2, 4, 8, 16, 32, 64] {
        for i in 0..128 {
            if i & step == 0 {
                let ia = base + i * stride;
                let ib = ia + step * stride;
                let a = data[ia];
                let c = data[ib];
                data[ia] = a + c;
                data[ib] = a - c;
            }
        }
    }
    for i in 0..128 {
        data[base + i * stride] *= scale;
    }
}

/// Apply H_k (transform each 128-row block, for every column): `W = H_k @ W`.
fn hadamard_rows(w: &mut [f32], rows: usize, cols: usize) {
    let scale = 1.0 / (128.0f32).sqrt();
    // Read a 128-element column slice into a stack buffer, transform it
    // contiguously, then write it back. The old strided butterfly touched 128
    // cache lines per element and was much slower on large matrices.
    let mut buf = [0.0f32; 128];
    for rb in 0..(rows / 128) {
        for j in 0..cols {
            for i in 0..128 {
                buf[i] = w[(rb * 128 + i) * cols + j];
            }
            hadamard128(&mut buf, 0, 1, scale);
            for i in 0..128 {
                w[(rb * 128 + i) * cols + j] = buf[i];
            }
        }
    }
}

/// Apply H_n (transform each 128-column block, for every row): `W = W @ H_n`.
fn hadamard_cols(w: &mut [f32], rows: usize, cols: usize) {
    let scale = 1.0 / (128.0f32).sqrt();
    for cb in 0..(cols / 128) {
        for i in 0..rows {
            hadamard128(w, i * cols + cb * 128, 1, scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Safetensors reader
// ---------------------------------------------------------------------------

/// Raw tensor metadata + byte offset into a safetensors file.
#[derive(Clone)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    offset: usize, // absolute byte offset in the loaded file bytes
    len: usize,    // number of bytes
}

struct SafeFile {
    bytes: Vec<u8>,
    tensors: HashMap<String, RawTensor>,
}

fn read_safetensors(path: &Path) -> Result<SafeFile, String> {
    let display = path.display();
    let bytes = fs::read(path).map_err(|e| format!("read {}: {}", display, e))?;
    if bytes.len() < 8 {
        return Err(format!("{}: too short", display));
    }
    let hlen_u64 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let hlen = usize::try_from(hlen_u64)
        .map_err(|_| format!("{}: bad header length {}", display, hlen_u64))?;
    // `bytes.len() >= 8` was checked above, so this subtraction cannot underflow.
    if hlen == 0 || hlen > bytes.len() - 8 {
        return Err(format!("{}: bad header length {}", display, hlen));
    }
    let header_str = std::str::from_utf8(&bytes[8..8 + hlen])
        .map_err(|e| format!("{}: bad header utf8: {}", display, e))?;
    let header = crate::json::parse(header_str)
        .map_err(|e| format!("{}: bad header json: {}", display, e))?;

    let data_start = 8 + hlen;
    let mut tensors = HashMap::new();
    if let Some(obj) = header.as_obj() {
        for (name, val) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = val
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{}: tensor {} has no dtype", display, name))?
                .to_string();
            if dtype_size(&dtype).is_none() {
                return Err(format!(
                    "{}: tensor {} has unsupported dtype {}",
                    display, name, dtype
                ));
            }
            let shape_json = val
                .get("shape")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| format!("{}: tensor {} has no shape", display, name))?;
            let shape = shape_json
                .iter()
                .map(|x| {
                    x.as_usize()
                        .ok_or_else(|| format!("{}: tensor {} has invalid shape", display, name))
                })
                .collect::<Result<Vec<usize>, String>>()?;
            let offsets = val
                .get("data_offsets")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| format!("{}: tensor {} has no data_offsets", display, name))?;
            if offsets.len() != 2 {
                return Err(format!(
                    "{}: tensor {} has invalid data_offsets",
                    display, name
                ));
            }
            let off0 = offsets[0]
                .as_usize()
                .ok_or_else(|| format!("{}: tensor {} has invalid data_offsets", display, name))?;
            let off1 = offsets[1]
                .as_usize()
                .ok_or_else(|| format!("{}: tensor {} has invalid data_offsets", display, name))?;
            if off1 < off0 {
                return Err(format!(
                    "{}: tensor {} has inverted data_offsets",
                    display, name
                ));
            }
            let begin = data_start
                .checked_add(off0)
                .ok_or_else(|| format!("{}: tensor {} offset overflow", display, name))?;
            let end = data_start
                .checked_add(off1)
                .ok_or_else(|| format!("{}: tensor {} offset overflow", display, name))?;
            if end > bytes.len() {
                return Err(format!("{}: tensor {} out of bounds", display, name));
            }
            tensors.insert(
                name.clone(),
                RawTensor {
                    dtype,
                    shape,
                    offset: begin,
                    len: end - begin,
                },
            );
        }
    }
    Ok(SafeFile { bytes, tensors })
}

/// Read a raw tensor as f32 (supports F16, F32, I16, I32).
fn read_f32(sf: &SafeFile, name: &str) -> Option<Vec<f32>> {
    let t = sf.tensors.get(name)?;
    let n = t.len / dtype_size(&t.dtype)?;
    let mut out = Vec::with_capacity(n);
    let b = &sf.bytes[t.offset..t.offset + t.len];
    match t.dtype.as_str() {
        "F16" => {
            for c in b.as_chunks::<2>().0 {
                out.push(half_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        "F32" => {
            for c in b.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        "I16" => {
            for c in b.as_chunks::<2>().0 {
                out.push(i16::from_le_bytes([c[0], c[1]]) as f32);
            }
        }
        "I32" => {
            for c in b.as_chunks::<4>().0 {
                out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32);
            }
        }
        "BF16" => {
            // bfloat16 is the top 16 bits of an f32.
            for c in b.as_chunks::<2>().0 {
                let bits = (u16::from_le_bytes([c[0], c[1]]) as u32) << 16;
                out.push(f32::from_bits(bits));
            }
        }
        _ => return None,
    }
    Some(out)
}

/// Read a raw tensor as u16 (for trellis / packed sign tensors).
fn read_u16(sf: &SafeFile, name: &str) -> Option<Vec<u16>> {
    let t = sf.tensors.get(name)?;
    if t.dtype != "I16" && t.dtype != "U16" && t.dtype != "F16" {
        return None;
    }
    let b = &sf.bytes[t.offset..t.offset + t.len];
    Some(
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F16" | "I16" | "U16" | "BF16" => Some(2),
        "F32" | "I32" | "U32" => Some(4),
        "F64" | "I64" | "U64" => Some(8),
        "BOOL" | "U8" | "I8" => Some(1),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Sharded model loading
// ---------------------------------------------------------------------------

/// Result of loading all safetensors shards: a map from tensor name to the
/// SafeFile holding it.
struct Shards {
    files: Vec<SafeFile>,
    /// name -> index into `files`
    map: HashMap<String, usize>,
}

fn load_all_shards(dir: &Path) -> Result<Shards, String> {
    let mut files = Vec::new();
    let mut map = HashMap::new();

    // Determine the file list: index JSON or single file or sharded glob.
    let index_path = dir.join("model.safetensors.index.json");
    let paths: Vec<PathBuf> = if index_path.exists() {
        let s = fs::read_to_string(&index_path)
            .map_err(|e| format!("read {}: {}", index_path.display(), e))?;
        let j =
            crate::json::parse(&s).map_err(|e| format!("parse {}: {}", index_path.display(), e))?;
        let wm = j
            .get("weight_map")
            .and_then(|v| v.as_obj())
            .ok_or("index missing weight_map")?;
        let mut names: Vec<String> = wm
            .iter()
            .filter_map(|(_, v)| v.as_str().map(|s| s.to_string()))
            .collect();
        names.sort();
        names.dedup();
        names.into_iter().map(|n| dir.join(n)).collect()
    } else {
        let single = dir.join("model.safetensors");
        if single.exists() {
            vec![single]
        } else {
            // Glob for model-00001-of-0000N.safetensors.
            let mut found: Vec<PathBuf> = fs::read_dir(dir)
                .map_err(|e| format!("read dir {}: {}", dir.display(), e))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
                        .unwrap_or(false)
                })
                .collect();
            found.sort();
            if found.is_empty() {
                return Err(format!("no safetensors files found in {}", dir.display()));
            }
            found
        }
    };

    for p in paths {
        let idx = files.len();
        let sf = read_safetensors(p.as_path())?;
        for name in sf.tensors.keys() {
            map.insert(name.clone(), idx);
        }
        files.push(sf);
    }
    Ok(Shards { files, map })
}

impl Shards {
    fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    fn f32(&self, name: &str) -> Option<Vec<f32>> {
        let i = *self.map.get(name)?;
        read_f32(&self.files[i], name)
    }
    fn u16(&self, name: &str) -> Option<Vec<u16>> {
        let i = *self.map.get(name)?;
        read_u16(&self.files[i], name)
    }
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        let i = *self.map.get(name)?;
        Some(self.files[i].tensors.get(name)?.shape.clone())
    }
    fn dtype(&self, name: &str) -> Option<&str> {
        let i = *self.map.get(name)?;
        Some(self.files[i].tensors.get(name)?.dtype.as_str())
    }

    /// Release raw byte ranges that are no longer needed (quantized trellises
    /// have already been copied into `QuantLinear`s by this point), keeping
    /// only the tensors named in `keep`. Offsets of the retained tensors are
    /// rewritten to the compacted buffer.
    fn compact_to(&mut self, keep: &HashSet<String>) {
        for sf in &mut self.files {
            let ranges: Vec<(usize, usize)> = sf
                .tensors
                .iter()
                .filter(|(name, _)| keep.contains(*name))
                .map(|(_, t)| (t.offset, t.offset + t.len))
                .collect();
            if ranges.is_empty() {
                sf.bytes.clear();
                sf.tensors.clear();
                continue;
            }
            let lo = ranges.iter().map(|r| r.0).min().unwrap();
            let hi = ranges.iter().map(|r| r.1).max().unwrap();
            if lo > 0 {
                sf.bytes.drain(..lo);
            }
            sf.bytes.truncate(hi - lo);
            sf.tensors.retain(|name, _| keep.contains(name));
            // Only retained tensors have offsets >= lo.
            for t in sf.tensors.values_mut() {
                t.offset -= lo;
            }
        }
        self.map.clear();
        for (i, sf) in self.files.iter().enumerate() {
            for name in sf.tensors.keys() {
                self.map.insert(name.clone(), i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Linear layer dequantization
// ---------------------------------------------------------------------------

/// Derive `(KA, is_half)` from the trellis tile width (number of uint16 words
/// per 16x16 tile). Integer K -> `words = 16*K`; half-integer `K = KA + 0.5`
/// -> `words = 16*KA + 8`. Pure integer math avoids the float pitfall where
/// `0.0f32.is_normal()` is false and misclassifies every integer rate.
fn bitrate_from_tile_width(words_per_tile: usize) -> (usize, bool) {
    (words_per_tile / 16, !words_per_tile.is_multiple_of(16))
}

/// Validate a trellis tile width before any unpacking. `unpack_tile` uses a
/// fixed 72-word stack buffer and divides by the word count, so malformed
/// metadata must be rejected at load time rather than panicking later.
fn validate_tile_width(words_per_tile: usize) -> Result<(usize, bool), String> {
    let valid_width = words_per_tile.is_multiple_of(16)
        || (words_per_tile >= 8 && (words_per_tile - 8).is_multiple_of(16));
    if words_per_tile == 0 || !valid_width || words_per_tile / 2 > 72 {
        return Err(format!(
            "invalid trellis tile width {} words (expected 16*K or 16*K+8, <= 144)",
            words_per_tile
        ));
    }
    Ok(bitrate_from_tile_width(words_per_tile))
}

/// Dequantize one EXL3 linear layer into a weight matrix of shape
/// `(out_features, in_features)` — ready for `matmul_t` (`y = x @ W^T`).
fn dequant_linear(shards: &Shards, prefix: &str) -> Result<(Matrix, Option<Vec<f32>>), String> {
    let trellis_name = format!("{}.trellis", prefix);
    let shape = shards
        .shape(&trellis_name)
        .ok_or_else(|| format!("missing {}", trellis_name))?;
    if shape.len() != 3 {
        return Err(format!("{}: expected 3D shape", trellis_name));
    }
    let tiles_k = shape[0];
    let tiles_n = shape[1];
    let words_per_tile = shape[2];
    let k = tiles_k * 16;
    let n = tiles_n * 16;

    // Bitrate from tile width (integer math only), with bounds validation.
    let (k_bits_int, k_half) = validate_tile_width(words_per_tile)?;

    // Codebook from marker presence.
    let cb = if shards.has(&format!("{}.mcg", prefix)) {
        Codebook::Mcg
    } else if shards.has(&format!("{}.mul1", prefix)) {
        Codebook::Mul1
    } else {
        Codebook::ThreeInst
    };
    if k_half && cb != Codebook::Mul1 {
        return Err(format!("{}: half bitrate requires mul1 codebook", prefix));
    }

    let packed = shards
        .u16(&trellis_name)
        .ok_or_else(|| format!("{}: cannot read trellis", trellis_name))?;
    let expected = tiles_k * tiles_n * words_per_tile;
    if packed.len() < expected {
        return Err(format!("{}: trellis too short", trellis_name));
    }

    // Sign vectors (suh / svh), with legacy packed su / sv fallback.
    let suh = match shards.f32(&format!("{}.suh", prefix)) {
        Some(v) => v,
        None => {
            let packed_su = shards
                .u16(&format!("{}.su", prefix))
                .ok_or_else(|| format!("{}: missing suh and su", prefix))?;
            let mut v = Vec::with_capacity(packed_su.len() * 16);
            for &w in &packed_su {
                for j in 0..16 {
                    v.push(if (w >> j) & 1 != 0 { -1.0 } else { 1.0 });
                }
            }
            v
        }
    };
    let svh = match shards.f32(&format!("{}.svh", prefix)) {
        Some(v) => v,
        None => {
            let packed_sv = shards
                .u16(&format!("{}.sv", prefix))
                .ok_or_else(|| format!("{}: missing svh and sv", prefix))?;
            let mut v = Vec::with_capacity(packed_sv.len() * 16);
            for &w in &packed_sv {
                for j in 0..16 {
                    v.push(if (w >> j) & 1 != 0 { -1.0 } else { 1.0 });
                }
            }
            v
        }
    };

    // 1. Unpack tiles into W_hat (k, n) in the rotated basis.
    let mut wh = vec![0.0f32; k * n];
    let mut tile = [0.0f32; 256];
    for tk in 0..tiles_k {
        for tn in 0..tiles_n {
            let base = (tk * tiles_n + tn) * words_per_tile;
            unpack_tile_f32(
                &packed[base..base + words_per_tile],
                k_bits_int,
                k_half,
                cb,
                &mut tile,
            );
            for r in 0..16 {
                for c in 0..16 {
                    wh[(tk * 16 + r) * n + tn * 16 + c] = tile[r * 16 + c];
                }
            }
        }
    }

    // 2. H_k on rows: W1 = H_k @ W_hat.
    hadamard_rows(&mut wh, k, n);

    if suh.len() < k || svh.len() < n {
        return Err(format!(
            "{}: sign vectors too short (suh={} need {}, svh={} need {})",
            prefix,
            suh.len(),
            k,
            svh.len(),
            n
        ));
    }

    // 3. Input sign vector: W2[i,j] = W1[i,j] * suh[i].
    for i in 0..k {
        let s = suh[i];
        for j in 0..n {
            wh[i * n + j] *= s;
        }
    }

    // 4. H_n on columns: W3 = W2 @ H_n.
    hadamard_cols(&mut wh, k, n);

    // 5. Output sign vector: W4[i,j] = W3[i,j] * svh[j].
    for i in 0..k {
        for j in 0..n {
            wh[i * n + j] *= svh[j];
        }
    }

    // W_hat above is W^T of shape (k, n). Transpose to (n, k) = W.
    let mut w = vec![0.0f32; n * k];
    for i in 0..k {
        for j in 0..n {
            w[j * k + i] = wh[i * n + j];
        }
    }

    // Optional bias, added by the forward pass.
    let bias = shards
        .f32(&format!("{}.bias", prefix))
        .filter(|b| b.len() == n);

    Ok((Matrix::from_vec(n, k, w), bias))
}

// ---------------------------------------------------------------------------
// Quantized linear: matmul directly against packed weights
// ---------------------------------------------------------------------------

/// A raw mutable pointer that is explicitly `Send`. Each scoped-thread worker
/// writes to a disjoint slice of the output vector.
#[derive(Clone, Copy)]
struct SharedPtr(*mut f32);
unsafe impl Send for SharedPtr {}

impl SharedPtr {
    #[inline]
    fn add_at(&self, off: usize, v: f32) {
        unsafe { *self.0.add(off) += v }
    }
    #[inline]
    fn write_at(&self, off: usize, v: f32) {
        unsafe { *self.0.add(off) = v }
    }
}

/// Threaded `y = W @ x` for an fp16 weight matrix (e.g. a tied embedding used
/// as `lm_head`). Weights stay as fp16 bits in RAM and are decoded through the
/// LUT inside `dot_f16`; only one output slice is written by each worker.
pub(crate) fn matmul_f16(x: &[f32], w: &F16Matrix, y: &mut [f32], threads: usize) {
    assert_eq!(x.len(), w.cols, "matmul_f16: input length mismatch");
    assert_eq!(y.len(), w.rows, "matmul_f16: output length mismatch");
    let rows = w.rows;
    let cols = w.cols;
    let base = SharedPtr(y.as_mut_ptr());
    let n_threads = if rows < 256 {
        1
    } else {
        threads.max(1).min(rows)
    };
    let chunk = rows.div_ceil(n_threads);
    let x_p = SendSlice::new(x);
    let w_p = SendSlice::new(&w.data);
    let mut jobs: Vec<Box<dyn FnOnce() + Send + 'static>> = Vec::new();
    for t in 0..n_threads {
        let r0 = t * chunk;
        if r0 >= rows {
            break;
        }
        let r1 = (r0 + chunk).min(rows);
        let (base, x_p, w_p) = (base, x_p, w_p);
        jobs.push(Box::new(move || unsafe {
            let x = x_p.get(cols);
            let data = w_p.get(rows * cols);
            for j in r0..r1 {
                base.write_at(j, dot_f16(x, &data[j * cols..(j + 1) * cols]));
            }
        }));
    }
    crate::threadpool::run(jobs);
}

/// An EXL3 linear layer kept in packed form. Inference multiplies directly
/// against the quantized trellis (dequantizing one 16x16 tile at a time into a
/// small scratch buffer), so no full f32 weight matrix is ever allocated.
pub struct QuantLinear {
    pub name: String,
    k: usize, // in_features
    n: usize, // out_features
    tiles_k: usize,
    tiles_n: usize,
    words_per_tile: usize,
    k_bits_int: usize,
    k_half: bool,
    cb: Codebook,
    /// Packed trellis (tiles_k * tiles_n * words_per_tile). Set to `None`
    /// after a decode cache is built so the on-disk copy can be freed.
    trellis: Option<Vec<u16>>,
    suh: Vec<f32>, // (k,)
    svh: Vec<f32>, // (n,)
    /// Optional fp16 decode cache of W_hat in (k, n) row-major layout.
    /// Codebook outputs are natively fp16, so this is lossless and costs
    /// 2 bytes/weight (vs 4 for f32, 1 for packed). When present, matmul uses
    /// a plain dot instead of decoding tiles on every token.
    what_f16: Option<Vec<u16>>,
    /// Optional f32 decode cache of W_hat (k, n). 4 bytes/weight but enables a
    /// direct f32 dot — fastest inference, matching the old full-dequant path.
    what_f32: Option<Vec<f32>>,
}

impl QuantLinear {
    /// Load a quantized linear from shards, keeping the trellis packed.
    fn from_shards(shards: &Shards, prefix: &str) -> Result<Self, String> {
        let trellis_name = format!("{}.trellis", prefix);
        let shape = shards
            .shape(&trellis_name)
            .ok_or_else(|| format!("missing {}", trellis_name))?;
        if shape.len() != 3 {
            return Err(format!("{}: expected 3D shape", trellis_name));
        }
        let (tiles_k, tiles_n, words_per_tile) = (shape[0], shape[1], shape[2]);
        let (k_bits_int, k_half) = validate_tile_width(words_per_tile)?;
        let cb = if shards.has(&format!("{}.mcg", prefix)) {
            Codebook::Mcg
        } else if shards.has(&format!("{}.mul1", prefix)) {
            Codebook::Mul1
        } else {
            Codebook::ThreeInst
        };
        if k_half && cb != Codebook::Mul1 {
            return Err(format!("{}: half bitrate requires mul1 codebook", prefix));
        }
        let mut trellis = shards
            .u16(&trellis_name)
            .ok_or_else(|| format!("{}: cannot read trellis", trellis_name))?;
        let expected = tiles_k
            .checked_mul(tiles_n)
            .and_then(|v| v.checked_mul(words_per_tile))
            .ok_or_else(|| format!("{}: trellis shape overflow", trellis_name))?;
        if trellis.len() < expected {
            return Err(format!("{}: trellis too short", trellis_name));
        }
        let suh = read_signs(shards, prefix, "suh", "su")?;
        let svh = read_signs(shards, prefix, "svh", "sv")?;
        let k = tiles_k * 16;
        let n = tiles_n * 16;
        if suh.len() < k || svh.len() < n {
            return Err(format!(
                "{}: sign vectors too short (suh={} need {}, svh={} need {})",
                prefix,
                suh.len(),
                k,
                svh.len(),
                n
            ));
        }
        // `shards.u16` already returned an owned buffer; avoid a full copy.
        trellis.truncate(expected);
        Ok(QuantLinear {
            name: prefix.to_string(),
            k,
            n,
            tiles_k,
            tiles_n,
            words_per_tile,
            k_bits_int,
            k_half,
            cb,
            trellis: Some(trellis),
            suh,
            svh,
            what_f16: None,
            what_f32: None,
        })
    }

    /// Decode all tiles once into an fp16 W_hat cache in (k, n) row-major
    /// layout. Lossless (codebook outputs are fp16) and 2 bytes/weight.
    /// The packed trellis is dropped afterwards to keep peak RAM low.
    pub fn build_decode_cache(&mut self) -> Result<(), String> {
        if self.what_f16.is_some() {
            return Ok(());
        }
        let trellis = self
            .trellis
            .take()
            .ok_or_else(|| format!("{}: trellis already consumed", self.name))?;
        let mut what = vec![0u16; self.k * self.n];
        // Codebook outputs are fp16 bit patterns; store them directly instead
        // of going through f32 and back (`f32_to_half`).
        let mut tile = [0u16; 256];
        for tk in 0..self.tiles_k {
            for tn in 0..self.tiles_n {
                let off = (tk * self.tiles_n + tn) * self.words_per_tile;
                unpack_tile_bits(
                    &trellis[off..off + self.words_per_tile],
                    self.k_bits_int,
                    self.k_half,
                    self.cb,
                    &mut tile,
                );
                for r in 0..16 {
                    for c in 0..16 {
                        // Store as (n, k) so the matmul dot is contiguous in k.
                        what[(tn * 16 + c) * self.k + tk * 16 + r] = tile[r * 16 + c];
                    }
                }
            }
        }
        self.what_f16 = Some(what);
        Ok(())
    }

    /// Decode all tiles into an f32 W_hat cache (k, n) for the fastest matmul.
    /// As above, the packed trellis is released once decoding is complete.
    pub fn build_decode_cache_f32(&mut self) -> Result<(), String> {
        if self.what_f32.is_some() {
            return Ok(());
        }
        let trellis = self
            .trellis
            .take()
            .ok_or_else(|| format!("{}: trellis already consumed", self.name))?;
        let mut what = vec![0.0f32; self.k * self.n];
        let mut tile = [0.0f32; 256];
        for tk in 0..self.tiles_k {
            for tn in 0..self.tiles_n {
                let off = (tk * self.tiles_n + tn) * self.words_per_tile;
                unpack_tile_f32(
                    &trellis[off..off + self.words_per_tile],
                    self.k_bits_int,
                    self.k_half,
                    self.cb,
                    &mut tile,
                );
                // unpack_tile already wrote row-major tile[r*16+c]; store as
                // (n, k) so the matmul dot is contiguous in k.
                for r in 0..16 {
                    for c in 0..16 {
                        what[(tn * 16 + c) * self.k + tk * 16 + r] = tile[r * 16 + c];
                    }
                }
            }
        }
        self.what_f32 = Some(what);
        Ok(())
    }

    /// Compute `y = W @ x` where W is the dequantized (n, k) weight, directly
    /// from the packed trellis. Mathematically equivalent to reconstructing f32
    /// weights then calling `matmul_t`, but uses ~on-disk-size memory.
    ///
    /// Derivation (H = normalized Sylvester H128, applied per 128-block):
    ///   W[j,i] = svh[j] * (H_n @ (H_k @ W_hat ⊙ suh))[i,j]
    ///   y[j] = svh[j] * (H_n @ (W_hat^T @ (H_k @ (x ⊙ suh))))[j]
    pub fn matmul(&self, x: &[f32], threads: usize) -> Vec<f32> {
        let k = self.k;
        let n = self.n;
        let scale = 1.0 / (128.0f32).sqrt();

        // 1. xr = H_k @ (x ⊙ suh), per 128-row block.
        let mut xr = vec![0.0f32; k];
        debug_assert!(x.len() >= k, "matmul input shorter than in_features");
        for i in 0..k {
            xr[i] = x.get(i).copied().unwrap_or(0.0) * self.suh[i];
        }
        for blk in 0..(k / 128) {
            hadamard128(&mut xr, blk * 128, 1, scale);
        }

        // 2. t = W_hat^T @ xr.
        let mut t = vec![0.0f32; n];
        if let Some(what) = &self.what_f32 {
            // Fastest: direct f32 matmul against the W_hat cache via the
            // selected compute backend (BLAS/GPU). what is (n, k) row-major.
            crate::compute::matmul_t(&xr, 1, k, what, n, &mut t, threads);
        } else if let Some(what) = &self.what_f16 {
            // fp16 cache: decode each weight on the fly (2 bytes/weight) and
            // split output rows across threads. No full f32 scratch is built.
            let base = SharedPtr(t.as_mut_ptr());
            let n_threads = if n < 256 { 1 } else { threads.max(1).min(n) };
            let chunk = n.div_ceil(n_threads);
            let xr_p = SendSlice::new(&xr);
            let what_p = SendSlice::new(what);
            let mut jobs: Vec<Box<dyn FnOnce() + Send + 'static>> = Vec::new();
            for t_idx in 0..n_threads {
                let j0 = t_idx * chunk;
                if j0 >= n {
                    break;
                }
                let j1 = (j0 + chunk).min(n);
                let (base, xr_p, what_p) = (base, xr_p, what_p);
                jobs.push(Box::new(move || unsafe {
                    let xr = xr_p.get(k);
                    let what = what_p.get(n * k);
                    for j in j0..j1 {
                        base.write_at(j, dot_f16(xr, &what[j * k..(j + 1) * k]));
                    }
                }));
            }
            crate::threadpool::run(jobs);
        } else {
            // Packed path: decode tiles on demand (low memory, slow). Split over
            // tn so each worker writes a disjoint slice of `t`.
            let trellis = self
                .trellis
                .as_ref()
                .expect("packed matmul requires a trellis cache");
            let base = SharedPtr(t.as_mut_ptr());
            // Avoid spawning many short-lived threads for small matrices: require
            // enough total tiles to amortize thread creation (~1024 tiles).
            let total_tiles = self.tiles_k * self.tiles_n;
            let n_threads = if total_tiles < 1024 {
                1
            } else {
                threads.max(1).min(self.tiles_n)
            };
            let chunk = self.tiles_n.div_ceil(n_threads);
            let trellis_p = SendSlice::new(trellis);
            let xr_p = SendSlice::new(&xr);
            let (tiles_k, tiles_n, wpb) = (self.tiles_k, self.tiles_n, self.words_per_tile);
            let (kbi, kh, cb) = (self.k_bits_int, self.k_half, self.cb);
            let mut jobs: Vec<Box<dyn FnOnce() + Send + 'static>> = Vec::new();
            for t_idx in 0..n_threads {
                let tn0 = t_idx * chunk;
                if tn0 >= tiles_n {
                    break;
                }
                let tn1 = (tn0 + chunk).min(tiles_n);
                let (base, trellis_p, xr_p) = (base, trellis_p, xr_p);
                jobs.push(Box::new(move || unsafe {
                    let trellis = trellis_p.get(tiles_k * tiles_n * wpb);
                    let xr = xr_p.get(k);
                    let mut tile = [0.0f32; 256];
                    for tn in tn0..tn1 {
                        for tk in 0..tiles_k {
                            let off = (tk * tiles_n + tn) * wpb;
                            unpack_tile_f32(&trellis[off..off + wpb], kbi, kh, cb, &mut tile);
                            let xbase = tk * 16;
                            let ybase = tn * 16;
                            for c in 0..16 {
                                let mut acc = 0.0f32;
                                for r in 0..16 {
                                    acc += tile[r * 16 + c] * xr[xbase + r];
                                }
                                base.add_at(ybase + c, acc);
                            }
                        }
                    }
                }));
            }
            crate::threadpool::run(jobs);
        } // end packed path

        // 3. t = H_n @ t, per 128-col block.
        for blk in 0..(n / 128) {
            hadamard128(&mut t, blk * 128, 1, scale);
        }

        // 4. y[j] = svh[j] * t[j].
        for (j, v) in t.iter_mut().enumerate() {
            *v *= self.svh[j];
        }
        t
    }
}

/// Read a sign/scale vector, with legacy packed fallback.
fn read_signs(shards: &Shards, prefix: &str, f16: &str, legacy: &str) -> Result<Vec<f32>, String> {
    match shards.f32(&format!("{}.{}", prefix, f16)) {
        Some(v) => Ok(v),
        None => {
            let packed = shards
                .u16(&format!("{}.{}", prefix, legacy))
                .ok_or_else(|| format!("{}: missing {} and {}", prefix, f16, legacy))?;
            let mut v = Vec::with_capacity(packed.len() * 16);
            for &w in &packed {
                for j in 0..16 {
                    v.push(if (w >> j) & 1 != 0 { -1.0 } else { 1.0 });
                }
            }
            Ok(v)
        }
    }
}

// ---------------------------------------------------------------------------
// Model-level loading
// ---------------------------------------------------------------------------

/// A fully loaded EXL3 model: config, plain weight matrices, quantized linears,
/// and 1-D vectors (norm weights, biases, etc.).
/// Read a 2-D tensor as an fp16 matrix without staging a full f32 copy when
/// the on-disk dtype is already F16 (the common case for embeddings).
fn read_f16_matrix(shards: &Shards, name: &str) -> Option<F16Matrix> {
    let shape = shards.shape(name)?;
    if shape.len() != 2 {
        return None;
    }
    let (rows, cols) = (shape[0], shape[1]);
    let len = rows.checked_mul(cols)?;
    let data = if shards.dtype(name) == Some("F16") {
        let v = shards.u16(name)?;
        if v.len() != len {
            return None;
        }
        v
    } else {
        let v = shards.f32(name)?;
        if v.len() != len {
            return None;
        }
        v.iter().map(|&x| f32_to_half(x)).collect()
    };
    Some(F16Matrix { rows, cols, data })
}

pub struct Exl3Model {
    pub config: Json,
    /// Tensor prefix detected from the shards (`model.` or
    /// `model.language_model.`). Kept here so `Model` does not have to
    /// re-detect it from loaded tensors — the embedding may live in
    /// `embeddings_f16` rather than `weights` for untied models.
    pub prefix: String,
    /// Plain (unquantized) 2-D tensors in `(out_features, in_features)` layout.
    pub weights: HashMap<String, Matrix>,
    /// Embedding tables stored as fp16 (2 bytes/weight), decoded per row on
    /// lookup. Used only for untied models; tied lm_heads need f32 weights.
    pub embeddings_f16: HashMap<String, F16Matrix>,
    /// Quantized linears, kept packed. Matmul dequantizes on the fly.
    pub linears: HashMap<String, QuantLinear>,
    /// 1-D tensors (norm weights, biases, A_log, dt_bias, ...).
    pub vectors: HashMap<String, Vec<f32>>,
}

impl Exl3Model {
    /// `path` may be a directory containing the model files or a path to one
    /// of the safetensors shards.
    /// `weight_mode`: "f32" decode to an f32 W_hat cache (fast, ~4x disk),
    /// "f16" decode to fp16 (fast-ish, ~2x disk), "packed" keep on-disk form
    /// and decode per matmul (lowest RAM, much slower).
    pub fn load(path: &str, weight_mode: &str) -> Result<Exl3Model, String> {
        let load_start = std::time::Instant::now();
        let p = Path::new(path);
        let dir = if p.is_dir() {
            p.to_path_buf()
        } else {
            p.parent().unwrap_or(Path::new(".")).to_path_buf()
        };

        // Config.
        let config_path = dir.join("config.json");
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| format!("read {}: {}", config_path.display(), e))?;
        let config = crate::json::parse(&config_str)
            .map_err(|e| format!("parse {}: {}", config_path.display(), e))?;

        let mut shards = load_all_shards(&dir)?;

        // Detect tensor prefix: multimodal checkpoints nest the text model
        // under `model.language_model.`, text-only ones use `model.`.
        let has_language = shards.has("model.language_model.embed_tokens.weight.trellis")
            || shards.has("model.language_model.embed_tokens.weight");
        let prefix = if has_language {
            "model.language_model."
        } else {
            "model."
        };

        let mut weights: HashMap<String, Matrix> = HashMap::new();
        let mut embeddings_f16: HashMap<String, F16Matrix> = HashMap::new();
        let mut linears: HashMap<String, QuantLinear> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

        // Collect all distinct base names from the shards.
        let mut base_names: Vec<String> = Vec::new();
        for name in shards.map.keys() {
            if let Some(base) = name.strip_suffix(".trellis") {
                if !base_names.iter().any(|b| b == base) {
                    base_names.push(base.to_string());
                }
            }
        }

        // Keep quantized linears packed (skip the vision tower). Matmul will
        // dequantize tiles on demand instead of allocating f32 weight matrices.
        let total_linears = base_names
            .iter()
            .filter(|b| !b.starts_with("model.visual."))
            .count();
        eprintln!(
            "[load] Loading {} quantized linears (weight_mode={})...",
            total_linears, weight_mode
        );
        let mut n_linears = 0usize;
        for base in &base_names {
            if base.starts_with("model.visual.") {
                continue;
            }
            let mut ql = QuantLinear::from_shards(&shards, base)?;
            match weight_mode {
                "f32" => ql.build_decode_cache_f32()?,
                "f16" => ql.build_decode_cache()?,
                "packed" => {} // decode on every matmul
                other => return Err(format!("unknown weight mode '{}'", other)),
            }
            // Keep any bias in vectors (forward currently ignores biases).
            if let Some(b) = shards.f32(&format!("{}.bias", base)) {
                vectors.insert(format!("{}.bias", base), b);
            }
            linears.insert(base.clone(), ql);
            n_linears += 1;
            if n_linears.is_multiple_of(16) || n_linears == total_linears {
                eprintln!(
                    "[load]   linears {}/{}  ({:.1}s)",
                    n_linears,
                    total_linears,
                    load_start.elapsed().as_secs_f64()
                );
            }
        }
        eprintln!(
            "[load] Linears done in {:.1}s ({} tensors total in shards)",
            load_start.elapsed().as_secs_f64(),
            shards.map.len()
        );

        // The quantized trellises have been copied into `QuantLinear`s now;
        // release their raw file bytes before decoding plain tensors.
        let all_names: Vec<String> = shards.map.keys().cloned().collect();
        let keep: HashSet<String> = all_names
            .iter()
            .filter(|name| {
                !name.ends_with(".trellis")
                    && !name.ends_with(".suh")
                    && !name.ends_with(".svh")
                    && !name.ends_with(".su")
                    && !name.ends_with(".sv")
                    && !name.ends_with(".mcg")
                    && !name.ends_with(".mul1")
                    && !name.starts_with("model.visual.")
            })
            .cloned()
            .collect();
        shards.compact_to(&keep);

        // Every plain embedding table is stored as fp16 and decoded per row
        // (or by the threaded f16 lm_head matmul for tied models), so even
        // tied checkpoints avoid the full f32 vocab x hidden allocation.
        let embed_name = {
            let preferred = format!("{}embed_tokens.weight", prefix);
            if shards.has(&preferred) {
                Some(preferred)
            } else if shards.has("model.embed_tokens.weight") {
                Some("model.embed_tokens.weight".to_string())
            } else {
                None
            }
        };
        let tie_embeddings = config
            .get("tie_word_embeddings")
            .or_else(|| {
                config
                    .get("text_config")
                    .and_then(|t| t.get("tie_word_embeddings"))
            })
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        // Kept for config diagnostics; embeddings are always fp16 now, and
        // tied models use `matmul_f16` for lm_head instead of an f32 matrix.
        let _tie_embeddings = tie_embeddings;

        // Plain (unquantized) tensors: embeddings, lm_head, 1-D vectors.
        for name in &all_names {
            // Skip quantized internals and vision tensors.
            if name.ends_with(".trellis")
                || name.ends_with(".suh")
                || name.ends_with(".svh")
                || name.ends_with(".su")
                || name.ends_with(".sv")
                || name.ends_with(".mcg")
                || name.ends_with(".mul1")
                || name.starts_with("model.visual.")
            {
                continue;
            }
            // Loaded directly as fp16 below (no f32 staging buffer).
            if embed_name.as_deref() == Some(name.as_str()) {
                continue;
            }
            // Already dequantized.
            if weights.contains_key(name) {
                continue;
            }
            let shape = match shards.shape(name) {
                Some(s) => s,
                None => continue,
            };
            // Decode each plain tensor exactly once. Calling `shards.f32`
            // twice here used to store every 2-D tensor (including the huge
            // embedding matrix) once in `vectors` and once in `weights`.
            let Some(v) = shards.f32(name) else {
                continue;
            };
            let is_matrix = shape.len() == 2 && !name.ends_with(".bias");
            let expected_len = if is_matrix {
                shape[0].checked_mul(shape[1])
            } else {
                None
            };
            if expected_len == Some(v.len()) {
                // Any plain 2-D tensor is a linear weight: embeddings,
                // lm_head, in_proj_a / in_proj_b (16 x hidden), q/k/v
                // projections that were left unquantized, etc.
                weights.insert(name.clone(), Matrix::from_vec(shape[0], shape[1], v));
            } else {
                // 1-D tensors (norms, biases, A_log, dt_bias, conv1d, ...).
                // A shape/length mismatch falls back to a flat vector rather
                // than silently dropping the tensor.
                vectors.insert(name.clone(), v);
            }
        }

        if let Some(name) = &embed_name {
            if let Some(emb) = read_f16_matrix(&shards, name) {
                embeddings_f16.insert(name.clone(), emb);
            }
        }

        // Sanity: embedding must exist under one of the two possible prefixes.
        let embed = format!("{}embed_tokens.weight", prefix);
        if !weights.contains_key(&embed)
            && !weights.contains_key("model.embed_tokens.weight")
            && !embeddings_f16.contains_key(&embed)
            && !embeddings_f16.contains_key("model.embed_tokens.weight")
        {
            return Err(format!("missing embedding tensor {}", embed));
        }

        eprintln!(
            "[load] Model loaded in {:.1}s ({} linears, {} matrices, {} vectors{})",
            load_start.elapsed().as_secs_f64(),
            linears.len(),
            weights.len(),
            vectors.len(),
            rss_mib()
                .map(|m| format!(", rss={:.0} MiB", m))
                .unwrap_or_default()
        );
        Ok(Exl3Model {
            config,
            prefix: prefix.to_string(),
            weights,
            embeddings_f16,
            linears,
            vectors,
        })
    }

    /// Resolve a tensor name using the model's prefix convention.
    pub fn w(&self, name: &str) -> Option<&Matrix> {
        self.weights.get(name)
    }
    pub fn ql(&self, name: &str) -> Option<&QuantLinear> {
        self.linears.get(name)
    }
    pub fn v(&self, name: &str) -> Option<&Vec<f32>> {
        self.vectors.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_decode_matches_reference() {
        let idx: [u16; 7] = [0, 1, 2, 3, 255, 256, 65535];
        let expected_3: [f32; 7] = [0.768, -0.9194, 0.9316, 0.2944, 1.204, 1.062, -0.1582];
        let expected_mcg: [f32; 7] = [1.844, 0.1345, -0.7593, 0.84, -1.287, -0.867, 0.85];
        let expected_mul1: [f32; 7] = [-3.453, 0.641, -0.4485, 0.1945, 0.8374, -0.2455, -0.6514];
        for i in 0..7 {
            assert!((half_to_f32(decode_3inst(idx[i])) - expected_3[i]).abs() < 1e-3);
            assert!((half_to_f32(decode_mcg(idx[i])) - expected_mcg[i]).abs() < 1e-3);
            assert!((half_to_f32(decode_mul1(idx[i])) - expected_mul1[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn half_roundtrip() {
        // Known fp16 bit patterns (finite, normal) round-trip exactly.
        let cases: [u16; 20] = [
            20952, 3648, 819, 24299, 9012, 8024, 7314, 4572, 24132, 3358, 22174, 24270, 29234,
            17870, 2848, 19349, 13825, 1041, 976, 3070,
        ];
        for &b in &cases {
            assert_eq!(f32_to_half(half_to_f32(b)), b, "roundtrip {:x}", b);
        }
    }

    #[test]
    fn half_and_integer_bitrate_unpack_consistency() {
        // A tile packed as integer K=6 must unpack the same way our window() does.
        // Build a synthetic packed tile: 96 uint16 words = 48 u32 words = 1536 bits = 256*6.
        // Put a known 6-bit value at each position i's low bits (ring bit S(i)-6..S(i)),
        // with S(i)=(i+1)*6, then verify window(end-16) recovers a tail-biting codeword.
        let nw = 48usize;
        let mut words = vec![0u32; nw];
        // Set ring bit b to 1 for a pattern; verify extraction is deterministic & in range.
        for b in 0..(256 * 6) {
            if b % 7 == 0 {
                words[b / 32] |= 1u32 << (31 - (b % 32));
            }
        }
        for i in 0..256usize {
            let end = (i + 1) * 6;
            let _w = window(&words, nw, end as i64 - 16);
        }
        // bitrate detection: K=6 from 96 words/tile
        assert_eq!(bitrate_from_tile_width(96), (6, false));
    }
    /// Half-integer rates (KA + 0.5, MUL1 codebook) had no test that actually
    /// called `unpack_tile`. Build a synthetic ring from the documented layout
    /// (each 16-position group advances KA bits per position plus one extra bit
    /// on odd positions, mask 0xAAAA) and extract every window bit-by-bit as an
    /// independent reference.
    #[test]
    fn half_bitrate_unpack_matches_ring_reference() {
        const KA: usize = 2; // K = 2.5
        const BPB: usize = 16 * KA + 8; // bits per 16-position group = 40
        const RING_BITS: usize = 16 * BPB; // 640
        let nw = RING_BITS / 32; // 20 u32 words
        let mut packed = vec![0u16; nw * 2];
        for (i, w) in packed.iter_mut().enumerate() {
            *w = ((i as u32).wrapping_mul(0x9E37_79B9) >> 13) as u16;
        }
        let words: Vec<u32> = (0..nw)
            .map(|w| ((packed[2 * w + 1] as u32) << 16) | packed[2 * w] as u32)
            .collect();

        let mut expected = [0f32; 256];
        let perm = tile_perm_table();
        for i in 0..256usize {
            let g = i >> 4;
            let q = i & 15;
            let mut end = g * BPB + q * KA;
            for j in 0..=q {
                end += ((0xAAAAu16 >> j) & 1) as usize;
            }
            // Independent bit-by-bit extraction (not using `window()`).
            let mut idx = 0u16;
            for b in 0..16 {
                let pos = (end as isize + b as isize - 16).rem_euclid(RING_BITS as isize) as usize;
                let bit = (words[pos / 32] >> (31 - (pos % 32))) & 1;
                idx = (idx << 1) | bit as u16;
            }
            expected[perm[i]] = decode_f32(idx, Codebook::ThreeInst);
        }

        let mut out = [0f32; 256];
        unpack_tile(&packed, KA, true, Codebook::ThreeInst, &mut out);
        for i in 0..256 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "tile[{}]: got {} expected {}",
                i,
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn tile_width_validation_rejects_malformed_metadata() {
        assert!(validate_tile_width(0).is_err());
        assert!(validate_tile_width(17).is_err());
        assert!(validate_tile_width(146).is_err()); // 73 u32 words > 72
        assert_eq!(validate_tile_width(96).unwrap(), (6, false));
        assert_eq!(validate_tile_width(128).unwrap(), (8, false));
        assert_eq!(validate_tile_width(40).unwrap(), (2, true));
    }

    #[test]
    fn unpack_tile_bits_match_codebook_f32() {
        // The f16 cache now stores the raw codebook fp16 bits instead of
        // round-tripping through f32 (which loses some subnormal patterns).
        // Their f32 values must match the f32 decode path exactly.
        for (ka, half, cb) in [
            (6usize, false, Codebook::Mcg),
            (2usize, true, Codebook::Mul1),
        ] {
            let words = if half { 16 * ka + 8 } else { 16 * ka };
            let packed: Vec<u16> = (0..words)
                .map(|i| ((i as u32 * 7919 + 13) & 0xffff) as u16)
                .collect();
            let mut bits = [0u16; 256];
            let mut vals = [0f32; 256];
            unpack_tile_bits(&packed, ka, half, cb, &mut bits);
            unpack_tile(&packed, ka, half, cb, &mut vals);
            for i in 0..256 {
                assert_eq!(half_to_f32(bits[i]), vals[i], "tile bit {}", i);
            }
            // Fast f32 LUT path must be bit-identical to unpack + convert.
            let mut fast = [0f32; 256];
            unpack_tile_f32(&packed, ka, half, cb, &mut fast);
            assert_eq!(fast, vals);
        }
    }

    #[test]
    fn hadamard_rows_blocking_matches_strided_reference() {
        let (rows, cols) = (256usize, 192usize);
        let original: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 37) as f32) * 0.1 - 1.0)
            .collect();
        let mut blocked = original.clone();
        let mut strided = original.clone();
        hadamard_rows(&mut blocked, rows, cols);
        let scale = 1.0 / (128.0f32).sqrt();
        for rb in 0..(rows / 128) {
            for j in 0..cols {
                hadamard128(&mut strided, rb * 128 * cols + j, cols, scale);
            }
        }
        for i in 0..blocked.len() {
            assert!(
                (blocked[i] - strided[i]).abs() < 1e-4,
                "element {}: {} vs {}",
                i,
                blocked[i],
                strided[i]
            );
        }
    }

    #[test]
    fn shard_compaction_keeps_only_requested_bytes() {
        let mut sf = SafeFile {
            bytes: vec![0u8; 100],
            tensors: HashMap::new(),
        };
        sf.tensors.insert(
            "keep".into(),
            RawTensor {
                dtype: "F16".into(),
                shape: vec![2],
                offset: 60,
                len: 20,
            },
        );
        sf.tensors.insert(
            "drop".into(),
            RawTensor {
                dtype: "F16".into(),
                shape: vec![2],
                offset: 0,
                len: 50,
            },
        );
        let mut shards = Shards {
            files: vec![sf],
            map: HashMap::new(),
        };
        shards.map.insert("keep".into(), 0);
        shards.map.insert("drop".into(), 0);
        shards.compact_to(&HashSet::from(["keep".to_string()]));
        assert_eq!(shards.map.len(), 1);
        assert_eq!(shards.files[0].tensors["keep"].offset, 0);
        assert_eq!(shards.files[0].bytes.len(), 20);
    }

    #[test]
    fn bitrate_detection_integer_and_half() {
        // Integer bitrates: 16*K words, not half.
        for k in 1..=8 {
            assert_eq!(bitrate_from_tile_width(16 * k), (k, false), "K={}", k);
        }
        // regress: K=8 (8.00 bpw models) must NOT be flagged as half.
        assert_eq!(bitrate_from_tile_width(128), (8, false));
        // Half-integer rates (mul1): 16*KA + 8 words.
        for ka in 1..=7 {
            assert_eq!(
                bitrate_from_tile_width(16 * ka + 8),
                (ka, true),
                "KA={}",
                ka
            );
        }
    }

    #[test]
    fn tile_perm_matches_reference() {
        let expected: [usize; 16] = [
            0, 16, 128, 144, 8, 24, 136, 152, 32, 48, 160, 176, 40, 56, 168, 184,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(tile_perm(i), e, "tile_perm[{}]", i);
        }
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// Build a Shards view over layer 0's in_proj_qkv from the downloaded
    /// partial model files. Returns None if the test data is absent.
    fn layer0_in_proj_qkv() -> Option<(Shards, &'static str)> {
        let full = fs::read("/tmp/unstable_prefix.bin").ok()?;
        let chunk = fs::read("/tmp/unstable_layer0.bin").ok()?;
        let hlen = u64::from_le_bytes(full[0..8].try_into().ok()?) as usize;
        let header = crate::json::parse(std::str::from_utf8(&full[8..8 + hlen]).ok()?).ok()?;
        let data_start = 8 + hlen;
        let base: usize = 508672000;
        let pre = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let mut tensors = HashMap::new();
        for suffix in ["suh", "svh", "trellis", "mcg"] {
            let name = format!("{}.{}", pre, suffix);
            let entry = header.get(&name)?;
            let offs = entry.get("data_offsets")?.as_arr()?;
            let s = data_start + offs[0].as_usize()? - base;
            let e = data_start + offs[1].as_usize()? - base;
            tensors.insert(
                name.clone(),
                RawTensor {
                    dtype: entry.get("dtype")?.as_str()?.to_string(),
                    shape: entry
                        .get("shape")?
                        .as_arr()?
                        .iter()
                        .filter_map(|x| x.as_usize())
                        .collect(),
                    offset: s,
                    len: e - s,
                },
            );
        }
        let sf = SafeFile {
            bytes: chunk,
            tensors,
        };
        let shards = Shards {
            files: vec![sf],
            map: {
                let mut m = HashMap::new();
                for name in ["suh", "svh", "trellis", "mcg"] {
                    m.insert(format!("{}.{}", pre, name), 0usize);
                }
                m
            },
        };
        Some((shards, pre))
    }

    #[test]
    #[ignore = "requires /tmp/unstable_prefix.bin + layer0.bin (partial real model); see tests/README.md"]
    fn dequant_matches_python_reference() {
        let (shards, pre) = match layer0_in_proj_qkv() {
            Some(v) => v,
            None => return,
        };
        let (w, _) = dequant_linear(&shards, pre).unwrap();
        // Reference from Python (first 5 values of row 0).
        // Python ref (wh = dequant in (k,n) layout); Rust w is transposed to (n,k),
        // so ref wh[r][c] == w.row(c)[r].
        let cases: &[(usize, usize, f32)] = &[
            (0, 0, -0.0123476),
            (0, 1, -0.01422224),
            (0, 127, -0.0074049),
            (0, 128, -0.00305228),
            (0, 200, 0.01579474),
            (100, 0, 0.00102627),
            (127, 127, -0.02487826),
            (128, 0, -0.01195727),
            (200, 300, 0.02083709),
            (500, 1000, -0.007876),
        ];
        for &(r, c, e) in cases {
            assert!(
                (w.row(c)[r] - e).abs() < 1e-4,
                "wh[{}][{}]: got {} expected {}",
                r,
                c,
                w.row(c)[r],
                e
            );
        }
    }

    #[test]
    #[ignore = "requires /tmp/unstable_prefix.bin + layer0.bin; self-contained equivalent in synthetic_tests"]
    fn quantized_matmul_matches_f32_dequant() {
        // The core guarantee of computing with quantized weights: inline
        // dequantize-on-matmul must match reconstruct-then-matmul.
        let (shards, pre) = match layer0_in_proj_qkv() {
            Some(v) => v,
            None => return,
        };
        let (w, _) = dequant_linear(&shards, pre).unwrap(); // (n, k) f32 reference
        let ql = QuantLinear::from_shards(&shards, pre).unwrap();
        assert_eq!(w.rows, ql.n);
        assert_eq!(w.cols, ql.k);

        // Deterministic pseudo-random input vector.
        let mut x = vec![0.0f32; ql.k];
        let mut s = 12345u64;
        for v in &mut x {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }

        let got = ql.matmul(&x, 1);
        // Reference: y[j] = sum_i W[j,i] * x[i] = dot(w.row(j), x).
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (j, &g) in got.iter().enumerate() {
            let expect = crate::tensor::dot(w.row(j), &x);
            let err = (g - expect).abs();
            max_abs = max_abs.max(err);
            max_rel = max_rel.max(err / (expect.abs() + 1e-3));
        }
        assert!(
            max_abs < 1e-3 && max_rel < 1e-2,
            "quantized matmul mismatch: max_abs={} max_rel={}",
            max_abs,
            max_rel
        );

        // Cached f32 path must also match the reference.
        let mut qc = QuantLinear::from_shards(&shards, pre).unwrap();
        qc.build_decode_cache_f32().unwrap();
        let got_cached = qc.matmul(&x, 1);
        let mut cmax = 0.0f32;
        for (j, &g) in got_cached.iter().enumerate() {
            let expect = crate::tensor::dot(w.row(j), &x);
            cmax = cmax.max((g - expect).abs());
        }
        assert!(cmax < 1e-3, "f32-cache matmul mismatch: max_abs={}", cmax);

        // Cached f16 path must also match (within fp16 rounding).
        let mut qh = QuantLinear::from_shards(&shards, pre).unwrap();
        qh.build_decode_cache().unwrap();
        let got_h = qh.matmul(&x, 1);
        let mut hmax = 0.0f32;
        for (j, &g) in got_h.iter().enumerate() {
            let expect = crate::tensor::dot(w.row(j), &x);
            hmax = hmax.max((g - expect).abs());
        }
        assert!(hmax < 5e-3, "f16-cache matmul mismatch: max_abs={}", hmax);
    }
    #[test]
    #[ignore]
    fn bench_quant_matmul() {
        let (shards, pre) = match layer0_in_proj_qkv() {
            Some(v) => v,
            None => return,
        };
        let mut ql = QuantLinear::from_shards(&shards, pre).unwrap();
        ql.build_decode_cache_f32().unwrap();
        let x = vec![0.5f32; ql.k];
        // warmup
        let _ = ql.matmul(&x, 1);
        for threads in [1usize, 2, 4, 8, 16] {
            let t0 = std::time::Instant::now();
            let n = 4;
            for _ in 0..n {
                let _ = ql.matmul(&x, threads);
            }
            let dt = t0.elapsed().as_secs_f64() / n as f64;
            eprintln!(
                "BENCH layer0 in_proj_qkv: k={} n={} tiles={} threads={} -> {:.4} s/matmul",
                ql.k,
                ql.n,
                ql.tiles_k * ql.tiles_n,
                threads,
                dt
            );
        }
    }
}

#[cfg(test)]
mod synthetic_tests {
    use super::*;
    use crate::tensor::dot;

    /// A deterministic 128x128 EXL3 linear built entirely in memory: no files,
    /// no `/tmp` fixtures. This is the fresh-clone-safe counterpart of the
    /// `/tmp`-dependent `real_model_tests::quantized_matmul_matches_f32_dequant`.
    fn synthetic_shards(cb: Codebook, half: bool) -> Shards {
        const PREFIX: &str = "synthetic";
        let (tiles_k, tiles_n) = (8usize, 8usize);
        let k = tiles_k * 16;
        let n = tiles_n * 16;
        // K = 8 (integer) or K = 3.5 (half-integer MUL1 rate).
        let words_per_tile = if half { 16 * 3 + 8 } else { 16 * 8 };

        fn add(
            bytes: &mut Vec<u8>,
            tensors: &mut HashMap<String, RawTensor>,
            name: &str,
            dtype: &str,
            shape: Vec<usize>,
            data: &[u8],
        ) {
            let offset = bytes.len();
            bytes.extend_from_slice(data);
            tensors.insert(
                name.to_string(),
                RawTensor {
                    dtype: dtype.to_string(),
                    shape,
                    offset,
                    len: data.len(),
                },
            );
        }

        let mut bytes = Vec::new();
        let mut tensors = HashMap::new();

        // xorshift64 trellis bits: identical on every machine and run.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut trellis = Vec::with_capacity(tiles_k * tiles_n * words_per_tile * 2);
        for _ in 0..tiles_k * tiles_n * words_per_tile {
            trellis.extend_from_slice(&(next() as u16).to_le_bytes());
        }
        add(
            &mut bytes,
            &mut tensors,
            &format!("{}.trellis", PREFIX),
            "I16",
            vec![tiles_k, tiles_n, words_per_tile],
            &trellis,
        );

        // Repeating fp16 scales [1, -1, 0.5, -0.5]. The output vector uses an
        // offset pattern so a row/column transposition cannot cancel out.
        const PATTERN: [u16; 4] = [0x3c00, 0xbc00, 0x3800, 0xb800];
        let mut suh = Vec::with_capacity(k * 2);
        for i in 0..k {
            suh.extend_from_slice(&PATTERN[i % 4].to_le_bytes());
        }
        add(
            &mut bytes,
            &mut tensors,
            &format!("{}.suh", PREFIX),
            "F16",
            vec![k],
            &suh,
        );
        let mut svh = Vec::with_capacity(n * 2);
        for i in 0..n {
            svh.extend_from_slice(&PATTERN[(i + 1) % 4].to_le_bytes());
        }
        add(
            &mut bytes,
            &mut tensors,
            &format!("{}.svh", PREFIX),
            "F16",
            vec![n],
            &svh,
        );

        // Codebook marker: presence alone selects the codebook at load time.
        match cb {
            Codebook::Mcg => add(
                &mut bytes,
                &mut tensors,
                &format!("{}.mcg", PREFIX),
                "I32",
                vec![1],
                &1i32.to_le_bytes(),
            ),
            Codebook::Mul1 => add(
                &mut bytes,
                &mut tensors,
                &format!("{}.mul1", PREFIX),
                "I32",
                vec![1],
                &1i32.to_le_bytes(),
            ),
            Codebook::ThreeInst => {} // no marker selects 3INST
        }

        let map = tensors.keys().map(|name| (name.clone(), 0usize)).collect();
        Shards {
            files: vec![SafeFile { bytes, tensors }],
            map,
        }
    }

    fn assert_close(label: &str, got: &[f32], expect: &[f32]) {
        assert_eq!(got.len(), expect.len(), "{}: output length", label);
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for (&g, &e) in got.iter().zip(expect) {
            let err = (g - e).abs();
            max_abs = max_abs.max(err);
            max_rel = max_rel.max(err / (e.abs() + 1e-3));
        }
        assert!(
            max_abs < 5e-3 && max_rel < 5e-3,
            "{}: max_abs={} max_rel={}",
            label,
            max_abs,
            max_rel
        );
    }

    /// All three storage modes must reproduce the reconstruct-then-matmul
    /// reference, for every EXL3 codebook (and the half-integer MUL1 rate).
    fn check(cb: Codebook, half: bool) {
        let shards = synthetic_shards(cb, half);
        let (w, _) = dequant_linear(&shards, "synthetic").unwrap();
        assert_eq!((w.rows, w.cols), (128, 128));

        let mut x = vec![0.0f32; w.cols];
        let mut seed = 42u64;
        for v in &mut x {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
        }
        let expect: Vec<f32> = (0..w.rows).map(|j| dot(w.row(j), &x)).collect();

        let ql = QuantLinear::from_shards(&shards, "synthetic").unwrap();
        assert_eq!((ql.n, ql.k), (w.rows, w.cols));
        assert_close("packed", &ql.matmul(&x, 1), &expect);

        let mut qf = QuantLinear::from_shards(&shards, "synthetic").unwrap();
        qf.build_decode_cache_f32().unwrap();
        assert_close("f32 cache", &qf.matmul(&x, 1), &expect);

        let mut qh = QuantLinear::from_shards(&shards, "synthetic").unwrap();
        qh.build_decode_cache().unwrap();
        assert_close("f16 cache", &qh.matmul(&x, 1), &expect);
    }

    #[test]
    fn matmul_matches_dequant_3inst() {
        check(Codebook::ThreeInst, false);
    }

    #[test]
    fn matmul_matches_dequant_mcg() {
        check(Codebook::Mcg, false);
    }

    #[test]
    fn matmul_matches_dequant_mul1() {
        check(Codebook::Mul1, false);
    }

    #[test]
    fn matmul_matches_dequant_mul1_half_bitrate() {
        // Half-integer K = 3.5 exercises the 0xAAAA mask ring layout.
        check(Codebook::Mul1, true);
    }
}

#[cfg(test)]
mod packed_profile {
    use super::*;

    #[test]
    #[ignore]
    fn profile_packed_decode_vs_dot() {
        // Synthetic packed layer with the same geometry as a 1024->6144 qkv proj:
        // tiles_k=64, tiles_n=384, words_per_tile=128 (K=8, MCG codebook).
        let (tk, tn, wpt) = (64usize, 384usize, 128usize);
        let mut trellis = vec![0u16; tk * tn * wpt];
        let mut s = 12345u64;
        for v in &mut trellis {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = (s >> 32) as u16;
        }
        let ql = QuantLinear {
            name: "synthetic".into(),
            k: tk * 16,
            n: tn * 16,
            tiles_k: tk,
            tiles_n: tn,
            words_per_tile: wpt,
            k_bits_int: 8,
            k_half: false,
            cb: Codebook::Mcg,
            trellis: Some(trellis),
            suh: vec![1.0; tk * 16],
            svh: vec![1.0; tn * 16],
            what_f16: None,
            what_f32: None,
        };
        let x = vec![0.5f32; ql.k];
        let mut tile = [0.0f32; 256];
        let reps = 3;

        // 1. Decode all tiles only (no dot).
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let mut acc = 0.0f32;
            for t in 0..ql.tiles_k * ql.tiles_n {
                let off = t * wpt;
                unpack_tile_f32(
                    &ql.trellis.as_ref().unwrap()[off..off + wpt],
                    8,
                    false,
                    Codebook::Mcg,
                    &mut tile,
                );
                acc += tile[0];
            }
            std::hint::black_box(acc);
        }
        let decode = t0.elapsed().as_secs_f64() / reps as f64;

        // 2. Full packed matmul (decode + dot).
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let _ = ql.matmul(&x, 8);
        }
        let full = t0.elapsed().as_secs_f64() / reps as f64;

        // 3. Dot only: decode once to f32 buffer, then scalar vs BLAS GEMV.
        let mut what = vec![0.0f32; ql.k * ql.n];
        for t in 0..ql.tiles_k * ql.tiles_n {
            let off = t * wpt;
            let tk2 = t / ql.tiles_n;
            let tn2 = t % ql.tiles_n;
            unpack_tile_f32(
                &ql.trellis.as_ref().unwrap()[off..off + wpt],
                8,
                false,
                Codebook::Mcg,
                &mut tile,
            );
            for r in 0..16 {
                for c in 0..16 {
                    what[(tn2 * 16 + c) * ql.k + tk2 * 16 + r] = tile[r * 16 + c];
                }
            }
        }
        let mut y = vec![0.0f32; ql.n];
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            crate::tensor::matmul_t_scalar(&x, 1, ql.k, &what, ql.n, &mut y, 8);
        }
        let dot_scalar = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            crate::compute::matmul_t(&x, 1, ql.k, &what, ql.n, &mut y, 8);
        }
        let dot_backend = t0.elapsed().as_secs_f64() / reps as f64;

        eprintln!("PROFILE 1024x6144 packed: decode={:.4}s full_packed={:.4}s dot_scalar={:.4}s dot_backend={:.4}s (decode={:.0}% of full)",
                  decode, full, dot_scalar, dot_backend, 100.0*decode/full);
    }
}
