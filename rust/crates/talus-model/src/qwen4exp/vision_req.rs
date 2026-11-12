//! S12-v: a `GENI` request's pictures (the SVE1 embeddings-file path), built as generate.cpp's GENI block
//! (12826-12918 at d3e539db) does: read the file's records, give every prompt cell its (t, h, w) M-RoPE position, point
//! each `<|image_pad|>` cell at its embedding row, and key each picture for the conversation cache (#2228). The ERR texts
//! are the C++ ones, byte for byte. The engine then hands the table, the rows and the keys to the C++ structures the prompt
//! path and the checkpoints read (thin setter), so the numerics are the same by construction.

use std::io::Read;

/// generate.cpp `fnv1a(data, n, h)` (FNV-1a 64, seedable).
pub fn fnv1a(data: &[u8], mut h: u64) -> u64 {
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}
pub const FNV_SEED: u64 = 1_469_598_103_934_665_603;

/// One picture key: its first cell and its hash (`ConversationImageKey`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PicKey {
    pub start: i64,
    pub hash: u64,
}

/// A GENI request's pictures, ready for the engine.
pub struct Pictures {
    /// the embedding rows of every record, back to back (n_embd f32 each)
    pub rows: Vec<f32>,
    /// per prompt cell: offset (in f32) of its row in `rows`, or -1 for a text cell
    pub row_of: Vec<i64>,
    /// (t, h, w) per cell for `cells` cells (the prompt, then text positions for every later cell)
    pub mrope: Vec<i32>,
    pub keys: Vec<PicKey>,
}

struct Rec {
    n: i64,
    nx: i64,
    ny: i64,
    off: usize,
}

/// Reads the SVE1 records of `path` (generate.cpp: magic 0x31455653, n >= 1, nx >= 1, ny >= 1, nx * ny == n, width == n_embd).
fn read_records(path: &str, n_embd: i64, rows: &mut Vec<f32>) -> Result<Vec<Rec>, String> {
    let mut f = std::fs::File::open(path).map_err(|_| format!("cannot open {path}"))?;
    let mut recs = Vec::new();
    loop {
        let mut hdr = [0u8; 20];
        // fread(hdr, 4, 5): 0 ints = the end; 1-4 ints = a bad file
        let mut got = 0usize;
        while got < 20 {
            match f.read(&mut hdr[got..]) {
                Ok(0) => break,
                Ok(k) => got += k,
                Err(_) => break,
            }
        }
        let ints = got / 4;
        if ints == 0 {
            break;
        }
        let h: Vec<i32> = (0..5)
            .map(|i| i32::from_le_bytes(hdr[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect();
        if ints != 5
            || h[0] != 0x3145_5653
            || h[1] < 1
            || h[2] < 1
            || h[3] < 1
            || (h[2] as i64) * (h[3] as i64) != h[1] as i64
            || h[4] != n_embd as i32
        {
            return Err(format!(
                "bad embeddings file (expected strata-vision records of width {n_embd})"
            ));
        }
        let off = rows.len();
        let cnt = h[1] as usize * h[4] as usize;
        let mut buf = vec![0u8; cnt * 4];
        let mut have = 0usize;
        while have < buf.len() {
            match f.read(&mut buf[have..]) {
                Ok(0) => break,
                Ok(k) => have += k,
                Err(_) => break,
            }
        }
        if have != buf.len() {
            return Err("short embeddings file".to_string());
        }
        rows.extend(
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
        );
        recs.push(Rec {
            n: h[1] as i64,
            nx: h[2] as i64,
            ny: h[3] as i64,
            off,
        });
    }
    Ok(recs)
}

/// One picture as the layout sees it: its grid and where its first row lives (an offset into the host rows, or an index for the
/// caller's device rows), and the hash input when no server key is given.
pub struct PicRec<'a> {
    pub n: i64,
    pub nx: i64,
    pub ny: i64,
    pub off: i64,
    /// the bytes hashed after the grid when the request has no (or a mismatched) img_keys list
    pub content: &'a [u8],
}

/// The layout of a GENI request (generate.cpp 12880-12972), shared by the SVE path ([`build`]) and the pixel path (D3b-2):
/// positions, each pad cell's row (`off + j * n_embd`), keys; the ERR texts name the source (`what`: "embeddings file" for SVE,
/// "image encoder" for pixels) only where generate.cpp's text does.
pub fn place(
    recs: &[PicRec],
    ids: &[i64],
    pad: i64,
    n_embd: i64,
    cells: i64,
    req_img_keys: &[u64],
    what: &str,
) -> Result<(Vec<i64>, Vec<i32>, Vec<PicKey>), String> {
    let n = ids.len() as i64;
    let mut mrope = vec![0i32; (cells.max(0) as usize) * 3];
    let mut row_of = vec![-1i64; n as usize];
    let mut keys = Vec::new();
    let put = |m: &mut Vec<i32>, c: i64, t: i64, h: i64, w: i64| {
        if c >= 0 && c < cells {
            let i = c as usize * 3;
            m[i] = t as i32;
            m[i + 1] = h as i32;
            m[i + 2] = w as i32;
        }
    };
    let (mut p, mut i, mut k) = (0i64, 0i64, 0usize);
    while i < n {
        if ids[i as usize] != pad {
            put(&mut mrope, i, p, p, p);
            p += 1;
            i += 1;
            continue;
        }
        if k >= recs.len() {
            return Err(format!("the prompt has more images than the {what}"));
        }
        let im = &recs[k];
        k += 1;
        // what the conversation cache compares: a picture is its grid and its embeddings (or the server's key, #2228)
        let grid: Vec<u8> = [im.n, im.nx, im.ny]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut h = fnv1a(&grid, FNV_SEED);
        if req_img_keys.len() == recs.len() {
            h = fnv1a(&req_img_keys[k - 1].to_le_bytes(), h);
        } else {
            h = fnv1a(im.content, h);
        }
        keys.push(PicKey { start: i, hash: h });
        for j in 0..im.n {
            if i + j >= n || ids[(i + j) as usize] != pad {
                return Err(format!(
                    "image {k} has {} rows but fewer <|image_pad|> tokens",
                    im.n
                ));
            }
        }
        for j in 0..im.n {
            let (y, x) = (j / im.nx, j % im.nx);
            put(&mut mrope, i + j, p, p + y, p + x);
            row_of[(i + j) as usize] = im.off + j * n_embd;
        }
        i += im.n;
        p += im.nx.max(im.ny);
    }
    if k != recs.len() {
        return Err(format!("the {what} has more images than the prompt"));
    }
    if n > 0 && ids[(n - 1) as usize] == pad {
        return Err("the prompt cannot end in an image".to_string());
    }
    for c in n..cells {
        put(&mut mrope, c, p + (c - n), p + (c - n), p + (c - n));
    }
    Ok((row_of, mrope, keys))
}

/// Builds a GENI request's pictures from its SVE1 file. `cells` = the M-RoPE table's length (mrope_host.size() / 3). Err = the
/// text after `ERR ` (the engine then resets the table to the identity and prints it, as generate.cpp).
pub fn build(
    path: &str,
    ids: &[i64],
    pad: i64,
    n_embd: i64,
    cells: i64,
    req_img_keys: &[u64],
) -> Result<Pictures, String> {
    let mut rows = Vec::new();
    let recs = read_records(path, n_embd, &mut rows)?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), rows.len() * 4) };
    let pr: Vec<PicRec> = recs
        .iter()
        .map(|r| PicRec {
            n: r.n,
            nx: r.nx,
            ny: r.ny,
            off: r.off as i64,
            content: &bytes[r.off * 4..(r.off + (r.n * n_embd) as usize) * 4],
        })
        .collect();
    let (row_of, mrope, keys) = place(
        &pr,
        ids,
        pad,
        n_embd,
        cells,
        req_img_keys,
        "embeddings file",
    )?;
    Ok(Pictures {
        rows,
        row_of,
        mrope,
        keys,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sve(dir: &std::path::Path, name: &str, recs: &[(i32, i32, i32)], n_embd: i32) -> String {
        let mut b = Vec::new();
        let mut v = 0f32;
        for &(n, nx, ny) in recs {
            for x in [0x3145_5653, n, nx, ny, n_embd] {
                b.extend_from_slice(&(x as i32).to_le_bytes());
            }
            for _ in 0..n * n_embd {
                b.extend_from_slice(&v.to_le_bytes());
                v += 1.0;
            }
        }
        let p = dir.join(name);
        std::fs::write(&p, b).unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn positions_rows_keys_and_errs_follow_generate_cpp() {
        let d = std::env::temp_dir().join(format!("s12v-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let pad = 9i64;
        // text 2, picture 2x3 (6 rows: nx = 3, ny = 2), text 2; 4 extra cells
        let one = sve(&d, "one.sve", &[(6, 3, 2)], 2);
        let ids = [1, 2, 9, 9, 9, 9, 9, 9, 3, 4];
        let pc = build(&one, &ids, pad, 2, 14, &[]).unwrap();
        // cells 0, 1: t=h=w=0,1; picture at p=2: (2, 2+y, 2+x); text after continues at 2 + max(3, 2) = 5
        let want: Vec<[i32; 3]> = vec![
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 3],
            [2, 2, 4],
            [2, 3, 2],
            [2, 3, 3],
            [2, 3, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
        ];
        assert_eq!(
            pc.mrope
                .chunks(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect::<Vec<_>>(),
            want
        );
        assert_eq!(pc.row_of, vec![-1, -1, 0, 2, 4, 6, 8, 10, -1, -1]);
        assert_eq!(pc.rows[pc.row_of[3] as usize], 2.0);
        assert_eq!(pc.keys.len(), 1);
        assert_eq!(pc.keys[0].start, 2);
        // with the server's key the hash is grid + key, not the floats
        let a = build(&one, &ids, pad, 2, 14, &[0xabc]).unwrap();
        let mut grid = Vec::new();
        for v in [6i64, 3, 2] {
            grid.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            a.keys[0].hash,
            fnv1a(&0xabcu64.to_le_bytes(), fnv1a(&grid, FNV_SEED))
        );
        assert_ne!(a.keys[0].hash, pc.keys[0].hash);
        // the ERR texts
        let e = |path: &str, ids: &[i64], w: i64| build(path, ids, pad, w, 32, &[]).err().unwrap();
        assert_eq!(
            e("/nonexistent/x.sve", &ids, 2),
            "cannot open /nonexistent/x.sve"
        );
        assert_eq!(
            e(&one, &ids, 3),
            "bad embeddings file (expected strata-vision records of width 3)"
        );
        assert_eq!(
            e(&one, &[1, 9, 9, 3], 2),
            "image 1 has 6 rows but fewer <|image_pad|> tokens"
        );
        assert_eq!(
            e(&one, &[1, 9, 9, 9, 9, 9, 9, 3, 9, 9, 3], 2),
            "the prompt has more images than the embeddings file"
        );
        let two = sve(&d, "two.sve", &[(6, 3, 2), (2, 1, 2)], 2);
        assert_eq!(
            e(&two, &ids, 2),
            "the embeddings file has more images than the prompt"
        );
        assert_eq!(
            e(&one, &[1, 9, 9, 9, 9, 9, 9], 2),
            "the prompt cannot end in an image"
        );
        let mut short = std::fs::read(&one).unwrap();
        short.truncate(short.len() - 4);
        std::fs::write(d.join("short.sve"), short).unwrap();
        assert_eq!(
            e(&d.join("short.sve").to_string_lossy(), &ids, 2),
            "short embeddings file"
        );
        // fread(hdr, 4, 5) on a 3-byte tail reads 0 whole ints: the end of the file, as C++ (no error); 7 bytes = 1 int: bad
        let mut tail3 = std::fs::read(&one).unwrap();
        tail3.extend_from_slice(&[1, 2, 3]);
        std::fs::write(d.join("tail3.sve"), tail3).unwrap();
        assert!(build(
            &d.join("tail3.sve").to_string_lossy(),
            &ids,
            pad,
            2,
            14,
            &[]
        )
        .is_ok());
        let mut tail7 = std::fs::read(&one).unwrap();
        tail7.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
        std::fs::write(d.join("tail7.sve"), tail7).unwrap();
        assert_eq!(
            e(&d.join("tail7.sve").to_string_lossy(), &ids, 2),
            "bad embeddings file (expected strata-vision records of width 2)"
        );
        std::fs::remove_dir_all(&d).ok();
    }
}
