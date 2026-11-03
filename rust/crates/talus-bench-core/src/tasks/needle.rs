//! `needle`: the long-context gate of docs/METHODOLOGY.md, reproducing `bench/needle.py` exactly:
//! - corpus: `sorted(glob("/usr/lib/python3.*/[a-m]*.py"))` read as text (invalid UTF-8 dropped, universal
//!   newlines) and joined with a blank line; its sha256 goes into the case's dataset revision, so runs on
//!   machines with a different corpus never pair;
//! - prompt: the server's `/tokenize` of the corpus cut to N tokens and `/detokenize`d, five modules and
//!   codenames drawn with Python's `random.Random(seed)` (Mersenne Twister ported below, bit-exact), each comment
//!   inserted at the line start before 90/70/50/30/10% of the text, then the question;
//! - request: max_tokens 4000, temperature 1.0, top_p 0.95, the seed, `reasoning_effort: low`;
//! - score: codenames found in the answer content / 5; the case passes at 5/5. The gate (all_pass) needs every
//!   case to pass.
//!
//! Params (`benchmarks.params.needle`): `lengths` (default smoke [32000], full [60000], qualification [60000,
//! 81000]), `seeds` (default smoke [1], else [1, 2, 3]), `corpus` (a file to use instead of the glob). A length
//! that does not fit the server's context with the 4000-token answer budget is left out and logged.

use super::{Task, TaskContext};
use crate::case::{Case, RawOutput, Score};
use crate::config::Tier;
use crate::datasets::DatasetPin;
use crate::util::sha256_hex;
use serde_json::{json, Value};

pub struct NeedleTask;

pub const WORDS: [&str; 20] = [
    "amber", "basalt", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo",
    "juniper", "kestrel", "lumen", "marble", "nimbus", "onyx", "pebble", "quartz", "raven",
    "sable", "tundra",
];
pub const MODULES: [&str; 8] = [
    "billing",
    "ingest",
    "scheduler",
    "auth",
    "search",
    "mailer",
    "cache",
    "export",
];
pub const DEPTHS: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
const ANSWER_BUDGET: u64 = 4000;

/// CPython's `random.Random` (MT19937 + init_by_array seeding for int seeds), enough for needle.py.
pub struct PyRandom {
    mt: [u32; 624],
    idx: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> Self {
        let mut r = PyRandom {
            mt: [0; 624],
            idx: 624,
        };
        let mut key: Vec<u32> = Vec::new();
        let mut n = seed;
        while n > 0 {
            key.push(n as u32);
            n >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        r.init_genrand(19_650_218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = 624.max(key.len());
        while k > 0 {
            let prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = 623;
        while k > 0 {
            let prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            k -= 1;
        }
        r.mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.idx = 624;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// getrandbits(k) for 0 < k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        self.genrand_u32() >> (32 - k)
    }

    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// _randbelow_with_getrandbits(n), n > 0.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        let k = 32 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    pub fn randint(&mut self, a: u32, b: u32) -> u32 {
        a + self.randbelow(b - a + 1)
    }

    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> &'a T {
        &seq[self.randbelow(seq.len() as u32) as usize]
    }

    /// random.sample for small populations (the pool algorithm CPython uses when n <= 21 for k <= 5).
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let n = population.len();
        assert!(
            k <= 5 && n <= 21 && k <= n,
            "only the small-population path is ported"
        );
        let mut pool = population.to_vec();
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let j = self.randbelow((n - i) as u32) as usize;
            out.push(pool[j].clone());
            pool[j] = pool[n - i - 1].clone();
        }
        out
    }
}

/// needle.py's draw for a seed: (module, codename) in the sampled order.
pub fn plan(seed: u64) -> Vec<(String, String)> {
    let mut rng = PyRandom::new(seed);
    let modules = rng.sample(&MODULES, 5);
    modules
        .into_iter()
        .map(|m| {
            let w = *rng.choice(&WORDS);
            let n = rng.randint(1000, 9999);
            (m.to_string(), format!("{w}-{n}"))
        })
        .collect()
}

/// Python `bytes.decode("utf-8", errors="ignore")`: drop invalid sequences.
fn decode_ignore(b: &[u8]) -> String {
    let mut out = String::with_capacity(b.len());
    let mut rest = b;
    loop {
        match std::str::from_utf8(rest) {
            Ok(s) => {
                out.push_str(s);
                return out;
            }
            Err(e) => {
                let (good, bad) = rest.split_at(e.valid_up_to());
                out.push_str(std::str::from_utf8(good).unwrap_or_default());
                let skip = e.error_len().unwrap_or(bad.len());
                rest = &bad[skip..];
            }
        }
    }
}

/// The methodology corpus (or the file named by params.corpus) and where it came from.
pub fn corpus(params: &Value) -> Result<(String, String), String> {
    if let Some(p) = params["corpus"].as_str() {
        let b = std::fs::read(p).map_err(|e| format!("needle corpus {p}: {e}"))?;
        return Ok((decode_ignore(&b), p.to_string()));
    }
    let mut files = Vec::new();
    let lib = std::path::Path::new("/usr/lib");
    let mut dirs: Vec<_> = std::fs::read_dir(lib)
        .map_err(|e| format!("/usr/lib: {e}"))?
        .flatten()
        .filter(|d| d.file_name().to_string_lossy().starts_with("python3."))
        .map(|d| d.path())
        .collect();
    dirs.sort();
    for d in dirs {
        for f in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let name = f.file_name().to_string_lossy().to_string();
            let first = name.chars().next().unwrap_or('z');
            if ('a'..='m').contains(&first) && name.ends_with(".py") && f.path().is_file() {
                files.push(f.path().display().to_string());
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err("needle corpus: no /usr/lib/python3.*/[a-m]*.py files (set benchmarks.params.needle.corpus)".into());
    }
    let mut parts = Vec::new();
    for f in &files {
        let t = decode_ignore(&std::fs::read(f).map_err(|e| format!("{f}: {e}"))?);
        // Python text mode: universal newlines
        parts.push(t.replace("\r\n", "\n").replace('\r', "\n"));
    }
    Ok((
        parts.join("\n\n"),
        format!("{} files under /usr/lib/python3.*", files.len()),
    ))
}

/// Insert the codename comments exactly as needle.py does (positions in characters).
pub fn build_doc(text: &str, plan: &[(String, String)]) -> String {
    let mut doc: Vec<char> = text.chars().collect();
    let mut order: Vec<(usize, f64)> = (0..plan.len()).map(|i| (i, DEPTHS[i])).collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, frac) in order {
        let end = (doc.len() as f64 * frac) as usize;
        let pos = doc[..end.min(doc.len())]
            .iter()
            .rposition(|c| *c == '\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        let note = format!(
            "# NOTE: the deployment codename for the {} module is {}.\n",
            plan[i].0, plan[i].1
        );
        doc.splice(pos..pos, note.chars());
    }
    doc.into_iter().collect()
}

pub fn question(plan: &[(String, String)]) -> String {
    let mods: Vec<&str> = plan.iter().map(|p| p.0.as_str()).collect();
    format!(
        "Somewhere in the code above are comments giving the deployment codename for these modules: {}. List each module with its codename, one per line as `module: codename`.",
        mods.join(", ")
    )
}

fn u64_list(v: &Value) -> Option<Vec<u64>> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
}

impl Task for NeedleTask {
    fn id(&self) -> &str {
        "needle"
    }

    fn scorer_version(&self) -> &'static str {
        "needle/1"
    }

    fn datasets(&self, _tier: Tier) -> Vec<&'static DatasetPin> {
        Vec::new()
    }

    fn needs_server(&self) -> bool {
        true
    }

    fn cases(&self, ctx: &TaskContext) -> Result<Vec<Case>, String> {
        let client = ctx.client.ok_or("needle: needs the server (tokenizer)")?;
        let lengths = u64_list(&ctx.params["lengths"]).unwrap_or(match ctx.tier {
            Tier::Smoke => vec![32_000],
            Tier::Full => vec![60_000],
            Tier::Qualification => vec![60_000, 81_000],
        });
        let seeds = u64_list(&ctx.params["seeds"]).unwrap_or(match ctx.tier {
            Tier::Smoke => vec![1],
            _ => vec![1, 2, 3],
        });
        // context of the served model, if the server says
        let ctx_max = client
            .get("/v1/talus/status")
            .ok()
            .filter(|r| r.status == 200)
            .and_then(|r| r.body["context"]["max"].as_u64())
            .filter(|m| *m > 0)
            .or_else(|| {
                client
                    .get("/props")
                    .ok()
                    .filter(|r| r.status == 200)
                    .and_then(|r| r.body["default_generation_settings"]["n_ctx"].as_u64())
            });
        let usable: Vec<u64> = lengths
            .iter()
            .copied()
            .filter(|n| ctx_max.is_none_or(|m| n + ANSWER_BUDGET <= m))
            .collect();
        if usable.is_empty() {
            return Err(format!(
                "needle: no length fits the served context {} (lengths {lengths:?} + {ANSWER_BUDGET} answer tokens)",
                ctx_max.unwrap_or(0)
            ));
        }
        let (text, source) = corpus(ctx.params)?;
        let corpus_sha = sha256_hex(text.as_bytes());
        let r = client
            .post("/tokenize", &json!({"content": text}))
            .map_err(|e| format!("needle: /tokenize: {e}"))?;
        if r.status != 200 {
            return Err(format!("needle: /tokenize answered {}", r.status));
        }
        let toks: Vec<Value> = r.body["tokens"]
            .as_array()
            .cloned()
            .ok_or("needle: /tokenize reply without tokens")?;
        let mut out = Vec::new();
        for &n in &usable {
            let cut: Vec<Value> = toks.iter().take(n as usize).cloned().collect();
            if (cut.len() as u64) < n {
                return Err(format!(
                    "needle: corpus has only {} tokens (< {n})",
                    toks.len()
                ));
            }
            let d = client
                .post("/detokenize", &json!({"tokens": cut}))
                .map_err(|e| format!("needle: /detokenize: {e}"))?;
            let prefix = d.body["content"]
                .as_str()
                .ok_or("needle: /detokenize reply without content")?
                .to_string();
            for &seed in &seeds {
                let p = plan(seed);
                let doc = build_doc(&prefix, &p);
                let q = question(&p);
                out.push(Case {
                    task: self.id().into(),
                    case_id: format!("needle/n{n}-s{seed}"),
                    dataset: "python-stdlib-corpus".into(),
                    dataset_revision: format!("corpus@{}", &corpus_sha[..12]),
                    messages: vec![json!({"role": "user", "content": format!("{doc}\n\n{q}")})],
                    expected: json!({"codes": p.iter().map(|(m, c)| json!([m, c])).collect::<Vec<_>>(), "n": n, "seed": seed,
                                      "depths": DEPTHS}),
                    meta: json!({"corpus": source}),
                    critical: false,
                    max_tokens: Some(ANSWER_BUDGET as u32),
                    tools: Value::Null,
                    request: json!({"seed": seed, "temperature": 1.0, "top_p": 0.95,
                                    "chat_template_kwargs": {"reasoning_effort": "low"}}),
                });
            }
        }
        Ok(super::take_limit(out, ctx.limit))
    }

    fn score(&self, case: &Case, out: &RawOutput) -> Score {
        let codes = case.expected["codes"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let mut missed = Vec::new();
        let mut hits = 0;
        for (i, mc) in codes.iter().enumerate() {
            let code = mc[1].as_str().unwrap_or("\u{0}");
            if out.content.contains(code) {
                hits += 1;
            } else {
                missed.push(format!(
                    "{}@{}",
                    mc[0].as_str().unwrap_or("?"),
                    DEPTHS.get(i).copied().unwrap_or(0.0)
                ));
            }
        }
        let of = codes.len().max(1);
        Score {
            pass: hits == codes.len() && !codes.is_empty(),
            value: hits as f64 / of as f64,
            parsed: json!({"found": hits, "of": codes.len()}),
            details: json!({"stage": if missed.is_empty() { "pass" } else { "missed" }, "missed": missed}),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_random_is_bit_exact() {
        let mut r = PyRandom::new(5);
        assert_eq!(
            [r.getrandbits(32), r.getrandbits(32), r.getrandbits(32)],
            [2675342405, 1097127993, 3185950873]
        );
        assert_eq!(r.random(), 0.35853551175589526);
        // needle.py draws (python3.12 reference values)
        let want: [(u64, [(&str, &str); 5]); 4] = [
            (
                1,
                [
                    ("scheduler", "pebble-8364"),
                    ("search", "pebble-7219"),
                    ("billing", "garnet-2537"),
                    ("export", "pebble-1464"),
                    ("mailer", "marble-8090"),
                ],
            ),
            (
                2,
                [
                    ("billing", "juniper-5121"),
                    ("export", "tundra-4476"),
                    ("cache", "tundra-1585"),
                    ("scheduler", "sable-3594"),
                    ("ingest", "nimbus-7447"),
                ],
            ),
            (
                3,
                [
                    ("auth", "tundra-8766"),
                    ("search", "sable-2073"),
                    ("cache", "tundra-1215"),
                    ("ingest", "pebble-5249"),
                    ("scheduler", "raven-4839"),
                ],
            ),
            (
                123456789,
                [
                    ("export", "juniper-8646"),
                    ("search", "marble-9764"),
                    ("mailer", "lumen-7547"),
                    ("cache", "onyx-9938"),
                    ("auth", "garnet-1387"),
                ],
            ),
        ];
        for (seed, exp) in want {
            let p = plan(seed);
            let got: Vec<(&str, &str)> = p.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            assert_eq!(got, exp.to_vec(), "seed {seed}");
        }
    }

    #[test]
    fn insertion_matches_needle_py() {
        // reference: needle.py's loop over the same text in python3.12 (non-ASCII text checks char positions)
        let text: String = (0..40).map(|i| format!("line{i:04}x é\n")).collect();
        let doc = build_doc(&text, &plan(1));
        let notes: Vec<usize> = doc
            .split('\n')
            .enumerate()
            .filter(|(_, l)| l.starts_with('#'))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(notes, vec![6, 18, 27, 35, 40]);
        assert_eq!(
            sha256_hex(doc.as_bytes()),
            "e5342b19f50625c10151a8613d2ebe83dc4d11fff8cfd4e4620988f34029dbcc"
        );
        assert!(question(&plan(1)).contains("scheduler, search, billing, export, mailer"));
    }

    #[test]
    fn decode_drops_invalid_bytes() {
        assert_eq!(decode_ignore(b"a\xffb\xc3\xa9"), "abé");
    }

    #[test]
    fn methodology_corpus_hash() {
        // reference (python3 -I, bench.py corpus()): python3.12 only -> 63 files, 1,701,119 chars, sha256 76a392fc...
        let pys: Vec<String> = std::fs::read_dir("/usr/lib")
            .map(|d| {
                d.flatten()
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .filter(|n| n.starts_with("python3."))
                    .collect()
            })
            .unwrap_or_default();
        if pys == ["python3.12"] {
            let (t, src) = corpus(&Value::Null).unwrap();
            assert_eq!(src, "63 files under /usr/lib/python3.*");
            assert_eq!(t.chars().count(), 1_701_119);
            assert_eq!(
                sha256_hex(t.as_bytes()),
                "76a392fc1b3be815de85a135560b7c3868103fb37b6bda1a8569fd152fc9cd38"
            );
        }
    }
}
