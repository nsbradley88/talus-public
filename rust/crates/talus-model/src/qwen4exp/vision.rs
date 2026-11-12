//! D3b-2: in-process image encoders for `--vision` (the interface frozen for the native encoder).
//!
//! A GENI request may carry its pictures as pixels instead of an SVE1 file:
//! `GENI <max_new> [keys] img_keys=<hex>,... img_px=<w>x<h>,... @px <ids>`, followed at once by a binary frame (the images'
//! RGB u8, row-major, back to back, exactly sum(w*h*3) bytes, no header). The engine hands the pixels to an [`ImageEncoder`],
//! which writes the embedding rows into device memory it owns; the prompt path copies them into `emb` per chunk (D3b-1's device
//! runs). talus-model does not depend on any encoder crate: talus-vision implements the trait, talus-cli wires it.

use core::ffi::c_void;

/// talus_hip::rt raw stream / event (backend-neutral `void*`).
pub type RawStream = *mut c_void;
pub type RawEvent = *mut c_void;

/// One picture of a request: its resized RGB u8 pixels and the server's key for it (`img_keys`, #2228).
#[derive(Clone, Copy, Debug)]
pub struct PixelImage<'a> {
    pub w: u32,
    pub h: u32,
    pub rgb: &'a [u8],
    pub key: u64,
}

/// One encoded picture: `n` = `nx * ny` rows of `n_embd` f32 in device memory, in merge order, valid until
/// [`ImageEncoder::release`].
#[derive(Clone, Copy, Debug)]
pub struct EncodedImage {
    pub rows: *const f32,
    pub n: i64,
    pub nx: i64,
    pub ny: i64,
}

pub trait ImageEncoder: Send {
    /// checked == the model's n_embd at load
    fn n_embd(&self) -> i64;
    /// rows per request; the engine refuses a request above it (ERR)
    fn max_rows(&self) -> i64;
    /// Enqueue the work on `stream`; returns rows valid until `release()`; records `done` after the last write.
    fn encode(
        &mut self,
        imgs: &[PixelImage],
        stream: RawStream,
        done: RawEvent,
    ) -> Result<Vec<EncodedImage>, String>;
    /// called by the engine after the request's prompt pass (also after an ERR or a cancelled read)
    fn release(&mut self);
}

/// Largest frame the engine accepts (the server's cap 4096 bound: 2048 x 2048 x 3).
pub const MAX_FRAME_BYTES: usize = 12 << 20;

/// The `img_px=<w>x<h>,...` list of a GENI line, and whether the line is the pixel form (`@px` where the SVE path stands).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PixelSpec {
    pub sizes: Vec<(u32, u32)>,
    /// sum(w * h * 3), the frame's exact length
    pub bytes: usize,
}

/// `img_px` grammar: `<w>x<h>` entries joined by ',', w and h decimal 1..=65535. None = malformed.
pub fn parse_img_px(v: &str) -> Option<PixelSpec> {
    let mut sizes = Vec::new();
    let mut bytes = 0usize;
    for e in v.split(',') {
        let (w, h) = e.split_once('x')?;
        if w.is_empty()
            || h.is_empty()
            || !w.bytes().all(|b| b.is_ascii_digit())
            || !h.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let (w, h): (u32, u32) = (w.parse().ok()?, h.parse().ok()?);
        if w == 0 || h == 0 || w > 65535 || h > 65535 {
            return None;
        }
        bytes = bytes.checked_add((w as usize).checked_mul(h as usize)?.checked_mul(3)?)?;
        sizes.push((w, h));
    }
    (!sizes.is_empty()).then_some(PixelSpec { sizes, bytes })
}

/// What the stdin reader must do after a protocol line: `Some(n)` = read exactly `n` raw bytes (the frame) before the next line.
/// Only a GENI line in the pixel form (`img_px=` key and an `@px` token) carries a frame; a malformed `img_px` = no frame can be
/// framed (the engine answers ERR and the client's bytes then arrive as lines: documented, the server never sends that).
pub fn frame_len(line: &str) -> Option<usize> {
    if !line.starts_with("GENI ") {
        return None;
    }
    let mut px = None;
    let mut at_px = false;
    for tok in line.split(' ') {
        if let Some(v) = tok.strip_prefix("img_px=") {
            px = Some(v);
        } else if tok == "@px" {
            at_px = true;
            break; // ids follow
        } else if !tok.contains('=') && tok != "GENI" && !tok.bytes().all(|b| b.is_ascii_digit()) {
            break; // an SVE path: no frame
        }
    }
    if !at_px {
        return None;
    }
    px.and_then(parse_img_px).map(|s| s.bytes)
}

/// The stdin reader's frame step: after `line`, read its frame (if it carries one) from `r`. Ok(None) = no frame; Ok(Some(v)) =
/// the frame (empty when over [`MAX_FRAME_BYTES`]: those bytes are read and dropped); Err = the input ended inside the frame.
#[allow(clippy::result_unit_err)]
pub fn read_frame(line: &str, r: &mut impl std::io::Read) -> Result<Option<Vec<u8>>, ()> {
    let Some(len) = frame_len(line) else {
        return Ok(None);
    };
    if len <= MAX_FRAME_BYTES {
        let mut buf = vec![0u8; len];
        r.read_exact(&mut buf).map_err(|_| ())?;
        Ok(Some(buf))
    } else {
        let n = std::io::copy(
            &mut std::io::Read::take(&mut *r, len as u64),
            &mut std::io::sink(),
        )
        .map_err(|_| ())?;
        if n == len as u64 {
            Ok(Some(Vec::new()))
        } else {
            Err(())
        }
    }
}

/// Test encoder (D3b-2 gate; `TALUS_VISION_MOCK=<dir>`): the rows of picture `key` are the SVE1 file `<dir>/<key as 16 hex>.sve`
/// (one record); the pixels are only checked for size. It copies the rows into device memory it owns, on the request's stream,
/// and records `done` there, so the engine path (event wait, device runs, release) is the real one. Not a shipping path.
pub struct MockEncoder {
    dir: std::path::PathBuf,
    n_embd: i64,
    max_rows: i64,
    dev: Vec<*mut c_void>,
    host: Vec<Vec<f32>>,
}

// SAFETY: the device pointers are only used from the serve loop's thread, between encode() and release().
unsafe impl Send for MockEncoder {}

impl MockEncoder {
    pub fn new(dir: impl Into<std::path::PathBuf>, n_embd: i64, max_rows: i64) -> Self {
        Self {
            dir: dir.into(),
            n_embd,
            max_rows,
            dev: Vec::new(),
            host: Vec::new(),
        }
    }

    /// One SVE1 record: (n, nx, ny, rows).
    fn read(&self, key: u64) -> Result<(i64, i64, i64, Vec<f32>), String> {
        let p = self.dir.join(format!("{key:016x}.sve"));
        let b = std::fs::read(&p).map_err(|e| format!("mock encoder: {}: {e}", p.display()))?;
        if b.len() < 20 {
            return Err(format!("mock encoder: {}: short", p.display()));
        }
        let h: Vec<i32> = (0..5)
            .map(|i| i32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();
        let (n, nx, ny, w) = (h[1] as i64, h[2] as i64, h[3] as i64, h[4] as i64);
        if h[0] != 0x3145_5653
            || w != self.n_embd
            || n != nx * ny
            || b.len() != 20 + (n * w) as usize * 4
        {
            return Err(format!(
                "mock encoder: {}: not one SVE1 record of width {}",
                p.display(),
                self.n_embd
            ));
        }
        Ok((
            n,
            nx,
            ny,
            b[20..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ))
    }
}

impl ImageEncoder for MockEncoder {
    fn n_embd(&self) -> i64 {
        self.n_embd
    }
    fn max_rows(&self) -> i64 {
        self.max_rows
    }
    fn encode(
        &mut self,
        imgs: &[PixelImage],
        stream: RawStream,
        done: RawEvent,
    ) -> Result<Vec<EncodedImage>, String> {
        self.release();
        let mut out = Vec::new();
        for im in imgs {
            if im.rgb.len() != im.w as usize * im.h as usize * 3 {
                return Err("mock encoder: pixel bytes do not match the size".into());
            }
            let (n, nx, ny, rows) = self.read(im.key)?;
            let bytes = rows.len() * 4;
            let mut d: *mut c_void = std::ptr::null_mut();
            // SAFETY: plain HIP runtime calls; `rows` stays alive in self.host until release() (the copy is async).
            unsafe {
                if hip_sys::hipMalloc(&mut d, bytes) != hip_sys::hipSuccess {
                    return Err("mock encoder: device allocation failed".into());
                }
                self.dev.push(d);
                if hip_sys::hipMemcpyAsync(
                    d,
                    rows.as_ptr().cast(),
                    bytes,
                    hip_sys::hipMemcpyHostToDevice,
                    stream as _,
                ) != hip_sys::hipSuccess
                {
                    return Err("mock encoder: upload failed".into());
                }
            }
            self.host.push(rows);
            out.push(EncodedImage {
                rows: d as *const f32,
                n,
                nx,
                ny,
            });
        }
        // SAFETY: as above
        if unsafe { hip_sys::hipEventRecord(done as _, stream as _) } != hip_sys::hipSuccess {
            return Err("mock encoder: event record failed".into());
        }
        Ok(out)
    }
    fn release(&mut self) {
        if self.dev.is_empty() {
            return;
        }
        // SAFETY: the engine calls release() after the prompt pass; free waits for the device to finish with the buffers
        unsafe {
            hip_sys::hipDeviceSynchronize();
            for d in self.dev.drain(..) {
                hip_sys::hipFree(d);
            }
        }
        self.host.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_takes_frames_raw_and_stays_in_sync() {
        use std::io::BufRead;
        // a pixel request whose frame bytes look like protocol lines, an oversize one, then a plain line
        let mut input = Vec::new();
        input.extend_from_slice(b"GENI 8 img_keys=1 img_px=2x2 @px 1,2\n");
        input.extend_from_slice(b"STOP\nQUIT\n"); // 10 of the 12 frame bytes
        input.extend_from_slice(b"ab");
        input.extend_from_slice(b"GENI 8 img_keys=1 img_px=2049x2048 @px 1\n");
        input.extend(std::iter::repeat_n(0u8, 2049 * 2048 * 3));
        input.extend_from_slice(b"GEN 4 1,2\n");
        let mut r = std::io::BufReader::new(&input[..]);
        let mut got = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            if r.read_line(&mut line).unwrap() == 0 {
                break;
            }
            let l = line.trim_end_matches('\n').to_string();
            let fr = read_frame(&l, &mut r).unwrap();
            got.push((l, fr.map(|f| f.len())));
        }
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].1, Some(12));
        assert_eq!(got[1].1, Some(0)); // over the cap: read and dropped
        assert_eq!(got[2], ("GEN 4 1,2".to_string(), None));
        // the input ends inside a frame: an error (the reader treats it as the end of input)
        let mut short = std::io::BufReader::new(&b"abc"[..]);
        assert!(read_frame("GENI 8 img_px=2x2 @px 1", &mut short).is_err());
    }

    #[test]
    fn img_px_and_frame_length() {
        assert_eq!(
            parse_img_px("4x2,1x1").unwrap(),
            PixelSpec {
                sizes: vec![(4, 2), (1, 1)],
                bytes: 27
            }
        );
        for bad in [
            "", "4x", "x2", "4x2,", "0x5", "4x2x1", "4X2", "-1x2", "65536x1", "4x2;1x1", "a x2",
        ] {
            assert!(parse_img_px(bad).is_none(), "{bad}");
        }
        assert_eq!(
            frame_len("GENI 8 img_keys=ab img_px=4x2 @px 1,2,3"),
            Some(24)
        );
        assert_eq!(
            frame_len("GENI 8 cache=0 img_keys=ab,cd img_px=4x2,2x2 @px 1,2"),
            Some(36)
        );
        assert_eq!(frame_len("GENI 8 /tmp/a.sve 1,2,3"), None); // SVE form
        assert_eq!(frame_len("GENI 8 img_px=4x2 /tmp/a.sve 1,2"), None); // path, not @px
        assert_eq!(frame_len("GEN 8 img_px=4x2 @px 1,2"), None);
        assert_eq!(frame_len("GENI 8 img_px=bad @px 1,2"), None); // malformed: no frame can be framed
        assert_eq!(frame_len("GENI 8 @px 1,2"), None); // no img_px: ERR, no frame
        assert_eq!(
            frame_len("GENI 8 img_px=2048x2048 @px 1"),
            Some(MAX_FRAME_BYTES)
        );
    }
}
