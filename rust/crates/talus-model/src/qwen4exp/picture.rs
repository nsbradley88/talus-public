//! D3b-1: picture rows for the batched prompt path (`--vision`).
//!
//! A picture's embedding rows replace the token embeddings of its `<|image_pad|>` cells. The source is either host memory
//! (the SVE file path: rows the C++ GENI loop read into its own buffer, one pointer per prompt cell) or device memory (an
//! in-process encoder's resident output slot, D3b-2). The prompt path asks [`PictureRows::chunk_runs`] for the maximal runs of
//! consecutive picture cells inside one chunk and copies each run into the chunk's `emb` rows:
//! - host rows: gathered into one pinned slot per run of at most 4 MiB and sent up as one copy (prefill.cpp 2462-2487, #1996),
//!   row by row from the caller's memory when no pinned slot can be had;
//! - device rows: one device-to-device copy per run (the encoder's rows are contiguous within an image).
//!
//! Bitwise contract: the bytes that reach `emb` are the source rows unchanged, in the same cells as the C++ path.

/// Where one prompt cell's picture row lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSrc {
    /// host memory, `n_embd` f32
    Host(*const f32),
    /// device memory, `n_embd` f32
    Device(*const f32),
}

/// One run of consecutive picture cells inside a chunk: `emb` rows `t0..t0+n` of the chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    /// first row of the run within the chunk
    pub t0: i64,
    /// rows in the run
    pub n: i64,
    /// source of the run's first row; the run's rows are contiguous in the source (`src + j * n_embd`) for a device run,
    /// and each host row is looked up per cell ([`PictureRows::host_row`]) as the C++ path does
    pub src: RowSrc,
}

/// D3b-2: the current request's device pictures, for the prompt path on this thread (the Rust serve loop sets them
/// around its prefill segments; the C++ structures never see device rows). (first cell, rows, device pointer of row 0).
pub struct DevicePictures {
    pub images: Vec<(i64, i64, *const f32)>,
    pub n_embd: i64,
}

thread_local! {
    static DEVICE_PICTURES: std::cell::RefCell<Option<DevicePictures>> = const { std::cell::RefCell::new(None) };
}

/// Sets (Some) or clears (None) this thread's device pictures; returns the previous value.
pub fn set_device_pictures(p: Option<DevicePictures>) -> Option<DevicePictures> {
    DEVICE_PICTURES.with(|c| c.replace(p))
}

/// Runs `f` with this thread's device pictures (None = none set).
pub fn with_device_pictures<R>(f: impl FnOnce(Option<&DevicePictures>) -> R) -> R {
    DEVICE_PICTURES.with(|c| f(c.borrow().as_ref()))
}

/// The picture rows of one prompt: per prompt cell (index relative to the prompt's first token), None for a text cell.
pub enum PictureRows<'a> {
    /// the C++ GENI loop's row table (`Prefill::embd_rows`): `n` entries, nullptr = text cell
    HostTable { rows: *const *const f32, n: i64 },
    /// device images in prompt order: (first cell, rows, device pointer of the first row), D3b-2
    Device {
        images: &'a [(i64, i64, *const f32)],
        n_embd: i64,
    },
}

/// Largest host run gathered into one pinned slot (prefill.cpp: 4 MiB of rows, at least one row).
pub fn host_run_max(n_embd: i64) -> i64 {
    let row_bytes = n_embd * 4;
    ((4i64 << 20) / row_bytes).max(1)
}

impl PictureRows<'_> {
    /// The picture row of prompt cell `i`, or None for a text cell.
    ///
    /// # Safety
    /// `HostTable.rows` must hold `n` valid entries.
    pub unsafe fn row(&self, i: i64) -> Option<RowSrc> {
        match self {
            PictureRows::HostTable { rows, n } => {
                if i < 0 || i >= *n {
                    return None;
                }
                let p = *rows.add(i as usize);
                (!p.is_null()).then_some(RowSrc::Host(p))
            }
            PictureRows::Device { images, n_embd } => images.iter().find_map(|&(c0, rows, src)| {
                (i >= c0 && i < c0 + rows)
                    .then(|| RowSrc::Device(src.add(((i - c0) * n_embd) as usize)))
            }),
        }
    }

    /// Whether any cell in `[c0, c0 + t)` is a picture cell (the chunk cannot take the batched token gather).
    ///
    /// # Safety
    /// As [`Self::row`].
    pub unsafe fn any_in(&self, c0: i64, t: i64) -> bool {
        (c0..c0 + t).any(|i| self.row(i).is_some())
    }

    /// The runs of picture cells in the chunk `[c0, c0 + t)`, in order. Host runs are cut at [`host_run_max`] rows (one
    /// pinned slot each, as the C++ path); device runs are cut where an image ends.
    ///
    /// # Safety
    /// As [`Self::row`].
    pub unsafe fn chunk_runs(&self, c0: i64, t: i64, n_embd: i64) -> Vec<Run> {
        let mut out = Vec::new();
        let host_max = host_run_max(n_embd);
        let mut j = 0i64;
        while j < t {
            let Some(first) = self.row(c0 + j) else {
                j += 1;
                continue;
            };
            let mut k = j + 1;
            match first {
                RowSrc::Host(_) => {
                    while k < t
                        && k - j < host_max
                        && matches!(self.row(c0 + k), Some(RowSrc::Host(_)))
                    {
                        k += 1;
                    }
                }
                RowSrc::Device(p0) => {
                    // contiguous in the encoder's slot: the next row is exactly one row further on
                    while k < t {
                        match self.row(c0 + k) {
                            Some(RowSrc::Device(p)) if p == p0.add(((k - j) * n_embd) as usize) => {
                                k += 1
                            }
                            _ => break,
                        }
                    }
                }
            }
            out.push(Run {
                t0: j,
                n: k - j,
                src: first,
            });
            j = k;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_runs_follow_the_cpp_rule() {
        let n_embd = 2560i64;
        let rows: Vec<f32> = vec![0.0; (n_embd * 1000) as usize];
        // cells: 3 text, 600 picture, 2 text, 5 picture
        let mut tab: Vec<*const f32> = vec![std::ptr::null(); 610];
        for i in 3..603 {
            tab[i] = rows.as_ptr().wrapping_add((i - 3) * n_embd as usize);
        }
        for i in 605..610 {
            tab[i] = rows
                .as_ptr()
                .wrapping_add((600 + i - 605) * n_embd as usize);
        }
        let p = PictureRows::HostTable {
            rows: tab.as_ptr(),
            n: tab.len() as i64,
        };
        assert_eq!(host_run_max(n_embd), 409);
        unsafe {
            assert!(!p.any_in(0, 3));
            assert!(p.any_in(0, 4));
            // whole prompt in one chunk: 600 rows cut at 409, then 191, then the second picture
            let r = p.chunk_runs(0, 610, n_embd);
            assert_eq!(
                r.iter().map(|r| (r.t0, r.n)).collect::<Vec<_>>(),
                vec![(3, 409), (412, 191), (605, 5)]
            );
            assert_eq!(r[0].src, RowSrc::Host(tab[3]));
            // a chunk boundary inside the picture: chunk 2 starts at cell 256 (row 0 of the chunk)
            let r = p.chunk_runs(256, 256, n_embd);
            assert_eq!(
                r.iter().map(|r| (r.t0, r.n)).collect::<Vec<_>>(),
                vec![(0, 256)]
            );
            assert_eq!(r[0].src, RowSrc::Host(tab[256]));
            // past the table: text
            assert!(p.row(610).is_none() && p.row(-1).is_none());
        }
    }

    #[test]
    fn device_pictures_are_per_thread() {
        let slot: Vec<f32> = vec![0.0; 8];
        let prev = set_device_pictures(Some(DevicePictures {
            images: vec![(3, 2, slot.as_ptr())],
            n_embd: 4,
        }));
        assert!(prev.is_none());
        assert_eq!(with_device_pictures(|p| p.map(|d| d.images.len())), Some(1));
        std::thread::spawn(|| assert!(with_device_pictures(|p| p.is_none())))
            .join()
            .unwrap();
        assert!(set_device_pictures(None).is_some());
        assert!(with_device_pictures(|p| p.is_none()));
    }

    #[test]
    fn device_runs_split_at_image_ends_and_chunks() {
        let n_embd = 4i64;
        let slot: Vec<f32> = vec![0.0; 64];
        let a = slot.as_ptr();
        let b = slot.as_ptr().wrapping_add(10 * n_embd as usize);
        // image A cells 2..8 (6 rows), image B cells 8..12 directly after (4 rows, not contiguous with A in the slot)
        let imgs = [(2i64, 6i64, a), (8, 4, b)];
        let p = PictureRows::Device {
            images: &imgs,
            n_embd,
        };
        unsafe {
            let r = p.chunk_runs(0, 16, n_embd);
            assert_eq!(
                r,
                vec![
                    Run {
                        t0: 2,
                        n: 6,
                        src: RowSrc::Device(a)
                    },
                    Run {
                        t0: 8,
                        n: 4,
                        src: RowSrc::Device(b)
                    },
                ]
            );
            // chunk [5, 10): rows 3.. of A, then B's first 2
            let r = p.chunk_runs(5, 5, n_embd);
            assert_eq!(
                r,
                vec![
                    Run {
                        t0: 0,
                        n: 3,
                        src: RowSrc::Device(a.wrapping_add(3 * 4))
                    },
                    Run {
                        t0: 3,
                        n: 2,
                        src: RowSrc::Device(b)
                    },
                ]
            );
        }
    }
}
