//! The five models and their options. Sources: docs/MODELS.md (files, shipped arguments, kill switches, opt-ins,
//! memory) and the release configuration (speculative sampling default on, Laguna verify GEMM and MiMo IQ1_M
//! opt-ins kept off after failing the quality gate).
//!
//! Only switches that exist for a model are listed. Exact speed-ups (kernel choices that leave the output
//! bit-identical) are not listed: they stay on at every level and there is no reason to turn them off outside
//! debugging.

use crate::{Choice, Edit, Group, ModelSpec, Quality, TuneOption};

const ON: &str = "on";
const OFF: &str = "off";
const SPEC_ON: Option<(&str, &str)> = Some(("spec", ON));

const fn toggle(off_edits: &'static [Edit]) -> [Choice; 2] {
    [
        Choice {
            value: ON,
            label: "On",
            edits: &[],
        },
        Choice {
            value: OFF,
            label: "Off",
            edits: off_edits,
        },
    ]
}

/// An opt-in: default off, `on_edits` turn it on.
const fn opt_in(on_edits: &'static [Edit]) -> [Choice; 2] {
    [
        Choice {
            value: OFF,
            label: "Off",
            edits: &[],
        },
        Choice {
            value: ON,
            label: "On",
            edits: on_edits,
        },
    ]
}

const CTX_DESC: &str = "The longest conversation (prompt plus answer) the model can hold.";
const CTX_TRADE: &str =
    "Shorter frees memory for other programs; longer chats are cut off or refused sooner.";

const SPEC_SAMPLING_DESC: &str =
    "For sampled requests (temperature > 0), drafts are accepted by probability, so more of them are kept.";
const SPEC_SAMPLING_TRADE: &str =
    "Same output distribution. Off only to reproduce older runs: a given seed gives different text.";

// Disk tier of the prompt cache (shared by the engines that have one). On by default (32 GiB, per-user folder).
const PCACHE_DISK_DEFAULT: &str = "32";
const PCACHE_DISK: [Choice; 3] = [
    Choice {
        value: "32",
        label: "On, up to 32 GiB",
        edits: &[],
    },
    Choice {
        value: "128",
        label: "On, up to 128 GiB",
        edits: &[Edit::Env("TALUS_PCACHE_DISK_GIB", "128")],
    },
    Choice {
        value: OFF,
        label: "Off",
        edits: &[Edit::Env("TALUS_PCACHE_DISK", "0")],
    },
];
const PCACHE_DISK_LABEL: &str = "Prompt cache on disk";
const PCACHE_DISK_DESC: &str =
    "Saves conversation state to disk, so after a restart or model switch a long chat resumes without re-reading.";
const PCACHE_DISK_SWITCH: &str =
    "TALUS_PCACHE_DISK=0 (off) | DIR, TALUS_PCACHE_DISK_GIB=N (default on: ~/.cache/talus/pcache, 32 GiB)";

// ------------------------------------------------------------------------------------------------- Qwen3.8-27B

// no --max-context = auto: the largest of 256K (native, no RoPE scaling) / 128K / 64K that fits in
// memory beside the weights and the int8 prompt weights (talus-cli logs the choice).
const Q27_CTX: [Choice; 6] = [
    Choice {
        value: "16384",
        label: "16K",
        edits: &[Edit::Arg("--max-context", "16384")],
    },
    Choice {
        value: "32768",
        label: "32K",
        edits: &[Edit::Arg("--max-context", "32768")],
    },
    Choice {
        value: "65536",
        label: "64K",
        edits: &[Edit::Arg("--max-context", "65536")],
    },
    Choice {
        value: "131072",
        label: "128K",
        edits: &[Edit::Arg("--max-context", "131072")],
    },
    Choice {
        value: "262144",
        label: "256K",
        edits: &[Edit::Arg("--max-context", "262144")],
    },
    Choice {
        value: "auto",
        label: "Auto (by memory, up to 256K)",
        edits: &[],
    },
];

const Q27_SPEC_OFF: &[Edit] = &[
    Edit::Remove("--mtp-gguf"),
    Edit::Remove("--mtp"),
    Edit::Remove("--spec-draft-vocab"),
    Edit::Remove("--ngram"),
    Edit::Remove("--dflash-gguf"),
    Edit::Remove("--drafter"),
];
const Q27_SPEC: [Choice; 2] = toggle(Q27_SPEC_OFF);

const Q27_DRAFTER: [Choice; 3] = [
    Choice {
        value: "auto",
        label: "Auto (by depth)",
        edits: &[],
    },
    Choice {
        value: "mtp",
        label: "MTP head only",
        edits: &[Edit::Arg("--drafter", "mtp")],
    },
    Choice {
        value: "dflash",
        label: "DFlash2 only",
        edits: &[Edit::Arg("--drafter", "dflash")],
    },
];

const Q27_NGRAM: [Choice; 2] = toggle(&[Edit::Remove("--ngram")]);
const Q27_PCACHE: [Choice; 2] = toggle(&[Edit::Arg("--prompt-cache", "0")]);
const Q27_NUM: [Choice; 2] = toggle(&[Edit::Env("TALUS_PF_FA", "0")]);
const Q27_I8: [Choice; 2] = toggle(&[Edit::Env("TALUS_PF_I8", "0")]);
const Q27_W4: [Choice; 2] = opt_in(&[Edit::Env("TALUS_Q27_W4I", "auto")]);
const Q27_W4FAST: [Choice; 2] = opt_in(&[Edit::Env("TALUS_Q27_W4I", "fast")]);

const Q27_OPTIONS: &[TuneOption] = &[
    TuneOption {
        id: "context",
        label: "Context length",
        description: CTX_DESC,
        tradeoff: CTX_TRADE,
        switch: "--max-context N",
        group: Group::Common,
        quality: Quality::Exact,
        default: "auto",
        choices: &Q27_CTX,
        requires: None,
    },
    TuneOption {
        id: "spec",
        label: "Speculative decoding",
        description: "Drafters guess the next few tokens and the model checks them all in one pass.",
        tradeoff: "Off: answers come out much slower. The output is identical either way.",
        switch: "drafter flags (--mtp-gguf, --mtp, --dflash-gguf, --ngram)",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &Q27_SPEC,
        requires: None,
    },
    TuneOption {
        id: "drafter",
        label: "Drafter",
        description: "Auto uses DFlash2 early in a conversation and the MTP head deeper in.",
        tradeoff: "Auto is fastest overall; a fixed drafter only fits chats that are always short or long.",
        switch: "--drafter auto|mtp|dflash",
        group: Group::Common,
        quality: Quality::Exact,
        default: "auto",
        choices: &Q27_DRAFTER,
        requires: SPEC_ON,
    },
    TuneOption {
        id: "ngram",
        label: "Repeat-text drafts",
        description: "Also drafts text that repeats earlier parts of the conversation (n-gram lookup).",
        tradeoff: "Helps most on edits and code that copy earlier text. Output is identical either way.",
        switch: "--ngram 24",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &Q27_NGRAM,
        requires: SPEC_ON,
    },
    TuneOption {
        id: "prompt_cache",
        label: "Prompt cache reuse",
        description: "Keeps snapshots of recent conversations so a follow-up reads only the new part.",
        tradeoff: "Off saves some GPU memory, but every turn re-reads the whole conversation.",
        switch: "--prompt-cache 8 (0 = off)",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &Q27_PCACHE,
        requires: None,
    },
    TuneOption {
        id: "prompt_cache_disk",
        label: PCACHE_DISK_LABEL,
        description: PCACHE_DISK_DESC,
        tradeoff: "Uses up to 32 or 128 GiB of disk and writes each long turn in the background. Output is identical.",
        switch: PCACHE_DISK_SWITCH,
        group: Group::Common,
        quality: Quality::Exact,
        default: PCACHE_DISK_DEFAULT,
        choices: &PCACHE_DISK,
        requires: Some(("prompt_cache", ON)),
    },
    TuneOption {
        id: "numerics",
        label: "Flash prompt attention",
        description: "Faster attention while reading prompts; rounds slightly differently from the old path.",
        tradeoff: "Off is the bit-for-bit reference path, with slower prompt reading.",
        switch: "TALUS_PF_FA=0 turns it off",
        group: Group::Accuracy,
        quality: Quality::Gated,
        default: ON,
        choices: &Q27_NUM,
        requires: None,
    },
    TuneOption {
        id: "w4_prompt",
        label: "4-bit prompt math (last 16 layers)",
        description: "Reads prompts with 4-bit math in the feed-forward of the last 16 layers: about 7% faster prompts.",
        tradeoff: "Passed our quality gate, but replies after a 2K prompt generate about 11% slower. Needs the 2 GiB .w4i file.",
        switch: "TALUS_Q27_W4I=auto",
        group: Group::Accuracy,
        quality: Quality::Gated,
        default: OFF,
        choices: &Q27_W4,
        requires: None,
    },
    TuneOption {
        id: "w4_fast",
        label: "4-bit prompt math (all layers)",
        description: "4-bit math in the feed-forward of every layer: about 30% faster prompts.",
        tradeoff: "Failed our quality gate: the output drifts further from the reference. Needs the 8 GiB .fast.w4i file. Off by default.",
        switch: "TALUS_Q27_W4I=fast",
        group: Group::Accuracy,
        quality: Quality::Reduced,
        default: OFF,
        choices: &Q27_W4FAST,
        requires: None,
    },
    TuneOption {
        id: "int8_prompt",
        label: "Int8 prompt weights",
        description: "Keeps an int8 copy of the weights for faster prompt reading (about 26 GiB more memory).",
        tradeoff: "Off frees about 26 GiB, but prompts read slower. Output is identical.",
        switch: "TALUS_PF_I8=0 turns it off",
        group: Group::Advanced,
        quality: Quality::Exact,
        default: ON,
        choices: &Q27_I8,
        requires: None,
    },
];

// ------------------------------------------------------------------------------------------- Qwen3.8-Flash-Next

const FN_CTX: [Choice; 4] = [
    Choice {
        value: "32768",
        label: "32K",
        edits: &[Edit::Arg("--max-context", "32768")],
    },
    Choice {
        value: "65536",
        label: "64K",
        edits: &[Edit::Arg("--max-context", "65536")],
    },
    Choice {
        value: "131072",
        label: "128K",
        edits: &[Edit::Arg("--max-context", "131072")],
    },
    Choice {
        value: "262144",
        label: "256K",
        edits: &[],
    },
];

// A native pack needs --spec T >= 2 (the verify window), so drafting stops by dropping the MTP runtime and the
// suffix drafter instead (`--spec 0` refuses to start).
const FN_SPEC: [Choice; 2] = toggle(&[
    Edit::Remove("--mtp"),
    Edit::Remove("--mtp-window"),
    Edit::Remove("--mtp-draft-vocab"),
    Edit::Arg("--suffix-draft", "0"),
]);
const FN_SUFFIX: [Choice; 2] = toggle(&[Edit::Arg("--suffix-draft", "0")]);
const FN_SSAMP: [Choice; 2] = toggle(&[Edit::Env("STRATA_SPEC_PROB", "0")]);
const FN_PCACHE: [Choice; 2] = toggle(&[Edit::Arg("--prompt-cache", "0")]);
const FN_VOCAB: [Choice; 2] = [
    Choice {
        value: "en",
        label: "English + code (faster code)",
        edits: &[],
    },
    Choice {
        value: "full",
        label: "Full (best for non-English, e.g. Chinese)",
        edits: &[Edit::Remove("--mtp-draft-vocab")],
    },
];
const FN_HC16: [Choice; 2] = opt_in(&[Edit::Env("TALUS_FN_HC_BF16", "2")]);
const FN_NUM: [Choice; 2] = toggle(&[
    Edit::Env("STRATA_PF_W16", "0"),
    Edit::Env("STRATA_PF_HCDOWN", "0"),
    Edit::Env("STRATA_MMVQ_KQ8", "0"),
    Edit::Env("STRATA_MMVQ_W32", "0"),
]);

const FN_OPTIONS: &[TuneOption] = &[
    TuneOption {
        id: "context",
        label: "Context length",
        description: CTX_DESC,
        tradeoff: CTX_TRADE,
        switch: "--max-context N",
        group: Group::Common,
        quality: Quality::Exact,
        default: "262144",
        choices: &FN_CTX,
        requires: None,
    },
    TuneOption {
        id: "spec",
        label: "Speculative decoding",
        description:
            "The MTP head guesses the next few tokens and the model checks them all in one pass.",
        tradeoff: "Off: answers come out much slower. The output is identical either way.",
        switch: "--mtp DIR (off: without --mtp, --suffix-draft 0)",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &FN_SPEC,
        requires: None,
    },
    TuneOption {
        id: "suffix",
        label: "Repeat-text drafts",
        description: "Also drafts continuations of text that appeared earlier in the conversation.",
        tradeoff:
            "Helps most on edits and code that copy earlier text. Output is identical either way.",
        switch: "--suffix-draft 3 (0 = MTP only)",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &FN_SUFFIX,
        requires: SPEC_ON,
    },
    TuneOption {
        id: "spec_sampling",
        label: "Speculative sampling",
        description: SPEC_SAMPLING_DESC,
        tradeoff: SPEC_SAMPLING_TRADE,
        switch: "STRATA_SPEC_PROB=0 turns it off",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &FN_SSAMP,
        requires: SPEC_ON,
    },
    TuneOption {
        id: "prompt_cache",
        label: "Prompt cache reuse",
        description: "Keeps conversation checkpoints so a follow-up reads only the new part.",
        tradeoff: "Off saves a little memory, but every turn re-reads the whole conversation.",
        switch: "--prompt-cache N (0 = off)",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &FN_PCACHE,
        requires: None,
    },
    TuneOption {
        id: "draft_vocab",
        label: "Draft vocabulary",
        description: "Which tokens the MTP drafter may propose. The English and code subset drafts faster.",
        tradeoff: "English: faster English and code, but Chinese text drafts poorly and runs slower. Same output.",
        switch: "--mtp-draft-vocab data/draft_vocab_en.bin (remove for the full vocabulary)",
        group: Group::Common,
        quality: Quality::Exact,
        default: "en",
        choices: &FN_VOCAB,
        requires: SPEC_ON,
    },
    TuneOption {
        id: "numerics",
        label: "Rounding-level speed-ups",
        description:
            "fp16 prompt experts, merged long-prompt projections and tuned decode layouts.",
        tradeoff: "Off is the exact reference path: slower prompts and decode.",
        switch: "STRATA_PF_W16=0 STRATA_PF_HCDOWN=0 STRATA_MMVQ_KQ8=0 STRATA_MMVQ_W32=0",
        group: Group::Accuracy,
        quality: Quality::Gated,
        default: ON,
        choices: &FN_NUM,
        requires: None,
    },
    TuneOption {
        id: "hc16",
        label: "16-bit HC streams",
        description: "Keeps the prompt path's hyper-connection streams in 16 bits: faster prompts.",
        tradeoff: "Failed our quality gate: the most likely next token changes far more often. Off by default.",
        switch: "TALUS_FN_HC_BF16=2",
        group: Group::Accuracy,
        quality: Quality::Reduced,
        default: OFF,
        choices: &FN_HC16,
        requires: None,
    },
];

// ------------------------------------------------------------------------------------------------ GLM-5.3-Flash

const GLM_CTX: [Choice; 3] = [
    Choice {
        value: "16384",
        label: "16K",
        edits: &[Edit::Arg("--ctx", "16384")],
    },
    Choice {
        value: "32768",
        label: "32K",
        edits: &[Edit::Arg("--ctx", "32768")],
    },
    Choice {
        value: "65536",
        label: "64K",
        edits: &[],
    },
];

const GLM_SPEC: [Choice; 2] = toggle(&[Edit::Flag("--no-mtp")]);
const GLM_PCACHE: [Choice; 2] = toggle(&[Edit::Env("GLM_PROMPT_SNAPS", "0")]);
const GLM_NUM: [Choice; 2] = toggle(&[
    Edit::Env("STRATA_PF_W16_MOE", "0"),
    Edit::Env("GLM_PCH", "2048"),
    Edit::Env("GLM_SPARSE_2P", "0"),
]);

const GLM_OPTIONS: &[TuneOption] = &[
    TuneOption {
        id: "context",
        label: "Context length",
        description: CTX_DESC,
        tradeoff: CTX_TRADE,
        switch: "--ctx N",
        group: Group::Common,
        quality: Quality::Exact,
        default: "65536",
        choices: &GLM_CTX,
        requires: None,
    },
    TuneOption {
        id: "spec",
        label: "Speculative decoding",
        description:
            "The model's built-in NextN layer drafts two tokens per round; the model checks them.",
        tradeoff: "Off: answers come out slower. The output is identical either way.",
        switch: "--no-mtp turns it off",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &GLM_SPEC,
        requires: None,
    },
    TuneOption {
        id: "prompt_cache",
        label: "Prompt cache reuse",
        description:
            "Keeps a snapshot of the conversation state so a follow-up reads only the new part.",
        tradeoff: "Off saves a little memory, but every turn re-reads the whole conversation.",
        switch: "GLM_PROMPT_SNAPS=0 turns it off",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &GLM_PCACHE,
        requires: None,
    },
    TuneOption {
        id: "numerics",
        label: "Rounding-level speed-ups",
        description: "fp16 routed experts, 4096-row chunks and two-pass sparse attention while reading prompts.",
        tradeoff: "Off is the earlier reference path, with slower prompt reading.",
        switch: "STRATA_PF_W16_MOE=0 GLM_PCH=2048 GLM_SPARSE_2P=0",
        group: Group::Accuracy,
        quality: Quality::Gated,
        default: ON,
        choices: &GLM_NUM,
        requires: None,
    },
];

// ---------------------------------------------------------------------------------------------- MiMo-V2.6-Flash

const MIMO_CTX: [Choice; 4] = [
    Choice {
        value: "32768",
        label: "32K",
        edits: &[Edit::Arg("--max-context", "32768")],
    },
    Choice {
        value: "65536",
        label: "64K",
        edits: &[Edit::Arg("--max-context", "65536")],
    },
    Choice {
        value: "131072",
        label: "128K",
        edits: &[Edit::Arg("--max-context", "131072")],
    },
    Choice {
        value: "196608",
        label: "192K",
        edits: &[],
    },
];

const MIMO_SPEC: [Choice; 2] = toggle(&[Edit::Flag("--no-mtp")]);
const MIMO_SSAMP: [Choice; 2] = toggle(&[Edit::Env("TALUS_MIMO_SSAMP", "0")]);
const MIMO_NUM: [Choice; 2] = toggle(&[
    Edit::Env("STRATA_IDS_W32", "0"),
    Edit::Env("TALUS_MIMO_FA", "0"),
]);
const MIMO_IQ1M: [Choice; 2] = opt_in(&[Edit::Env("MIMO_IQ1M_TPMOE", "1")]);

const MIMO_OPTIONS: &[TuneOption] = &[
    TuneOption {
        id: "context",
        label: "Context length",
        description: CTX_DESC,
        tradeoff: CTX_TRADE,
        switch: "--max-context N",
        group: Group::Common,
        quality: Quality::Exact,
        default: "196608",
        choices: &MIMO_CTX,
        requires: None,
    },
    TuneOption {
        id: "spec",
        label: "Speculative decoding",
        description:
            "The DFlash drafter guesses up to 4 tokens; the model checks them all in one pass.",
        tradeoff: "Off: answers come out much slower. The output is identical either way.",
        switch: "--dflash-gguf FILE --spec 4 (--no-mtp turns it off)",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &MIMO_SPEC,
        requires: None,
    },
    TuneOption {
        id: "spec_sampling",
        label: "Speculative sampling",
        description: SPEC_SAMPLING_DESC,
        tradeoff: SPEC_SAMPLING_TRADE,
        switch: "TALUS_MIMO_SSAMP=0 turns it off",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &MIMO_SSAMP,
        requires: SPEC_ON,
    },
    TuneOption {
        id: "numerics",
        label: "Rounding-level speed-ups",
        description:
            "Wave32 routed-expert mat-vecs and WMMA flash attention while reading prompts.",
        tradeoff: "Off is the exact reference path: slower prompts and decode.",
        switch: "STRATA_IDS_W32=0 TALUS_MIMO_FA=0",
        group: Group::Accuracy,
        quality: Quality::Gated,
        default: ON,
        choices: &MIMO_NUM,
        requires: None,
    },
    TuneOption {
        id: "iq1m",
        label: "Low-bit expert prompt kernel",
        description: "Reads prompts with a faster low-bit kernel for the routed experts.",
        tradeoff:
            "Changes the output measurably (it did not pass our quality gate). Off by default.",
        switch: "MIMO_IQ1M_TPMOE=1",
        group: Group::Accuracy,
        quality: Quality::Reduced,
        default: OFF,
        choices: &MIMO_IQ1M,
        requires: None,
    },
];

// ------------------------------------------------------------------------------------------------- Laguna-S-2.1

const LAG_CTX: [Choice; 4] = [
    Choice {
        value: "32768",
        label: "32K",
        edits: &[Edit::Arg("--max-context", "32768")],
    },
    Choice {
        value: "65536",
        label: "64K",
        edits: &[Edit::Arg("--max-context", "65536")],
    },
    Choice {
        value: "131072",
        label: "128K",
        edits: &[Edit::Arg("--max-context", "131072")],
    },
    Choice {
        value: "262144",
        label: "256K",
        edits: &[],
    },
];

// --spec 0 only stops prompt lookup; the DFlash drafter (from --dflash-gguf, TALUS_LAGUNA_DFLASH or a
// *dflash*Q8_0*.gguf found next to the model) keeps drafting unless TALUS_LAGUNA_DFLASH=0 and no --dflash-gguf.
const LAG_SPEC: [Choice; 2] = toggle(&[
    Edit::Arg("--spec", "0"),
    Edit::Remove("--dflash-gguf"),
    Edit::Env("TALUS_LAGUNA_DFLASH", "0"),
]);
const LAG_SSAMP: [Choice; 2] = toggle(&[Edit::Env("TALUS_LAGUNA_SSAMP", "0")]);
const LAG_SPARSE: [Choice; 2] = toggle(&[Edit::Remove("--sparse")]);
const LAG_LSRV: [Choice; 2] = toggle(&[Edit::Env("TALUS_LAGUNA_LSRV", "0")]);
const LAG_VGEMM: [Choice; 2] = opt_in(&[Edit::Env("TALUS_LAGUNA_VERIFY_GEMM", "9")]);

const LAG_OPTIONS: &[TuneOption] = &[
    TuneOption {
        id: "context",
        label: "Context length",
        description: CTX_DESC,
        tradeoff: CTX_TRADE,
        switch: "--max-context N",
        group: Group::Common,
        quality: Quality::Exact,
        default: "262144",
        choices: &LAG_CTX,
        requires: None,
    },
    TuneOption {
        id: "spec",
        label: "Speculative decoding",
        description: "Prompt lookup and the DFlash drafter guess ahead; the model checks the guesses in one pass.",
        tradeoff: "Off: answers come out much slower, edits most of all. Output is identical either way.",
        switch: "--spec 7, DFlash drafter (off: --spec 0, TALUS_LAGUNA_DFLASH=0)",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &LAG_SPEC,
        requires: None,
    },
    TuneOption {
        id: "spec_sampling",
        label: "Speculative sampling",
        description: SPEC_SAMPLING_DESC,
        tradeoff: SPEC_SAMPLING_TRADE,
        switch: "TALUS_LAGUNA_SSAMP=0 turns it off",
        group: Group::Common,
        quality: Quality::Exact,
        default: ON,
        choices: &LAG_SSAMP,
        requires: SPEC_ON,
    },
    TuneOption {
        id: "sparse",
        label: "Sparse attention past 32K",
        description: "Past 32K tokens, decode looks at the start, the recent text and the most relevant blocks.",
        tradeoff: "Off: full attention everywhere (the exact reference), slower in very long chats.",
        switch: "--sparse 1024,16384,128,1024,16 (remove to turn off)",
        group: Group::Accuracy,
        quality: Quality::Gated,
        default: ON,
        choices: &LAG_SPARSE,
        requires: None,
    },
    TuneOption {
        id: "served_speed",
        label: "Served-speed mode",
        description: "The tuned serving path: decode through the verify kernels and wave32 dense mat-vecs.",
        tradeoff: "Off restores the earlier reference path: exact reference numerics, clearly slower.",
        switch: "TALUS_LAGUNA_LSRV=0 turns it off",
        group: Group::Accuracy,
        quality: Quality::Gated,
        default: ON,
        choices: &LAG_LSRV,
        requires: None,
    },
    TuneOption {
        id: "verify_gemm",
        label: "Batched verify",
        description: "Checks drafts with one batched matrix multiply: much faster edits.",
        tradeoff: "Failed our quality gate: outputs change, and edits dropped doc comments in tests. Off by default.",
        switch: "TALUS_LAGUNA_VERIFY_GEMM=9",
        group: Group::Accuracy,
        quality: Quality::Reduced,
        default: OFF,
        choices: &LAG_VGEMM,
        requires: None,
    },
    TuneOption {
        id: "prompt_cache_disk",
        label: PCACHE_DISK_LABEL,
        description: PCACHE_DISK_DESC,
        tradeoff: "Uses up to 32 or 128 GiB of disk. Greedy output is identical; the drafter starts cold after a restore.",
        switch: PCACHE_DISK_SWITCH,
        group: Group::Common,
        quality: Quality::Exact,
        default: PCACHE_DISK_DEFAULT,
        choices: &PCACHE_DISK,
        requires: None,
    },
];

// ----------------------------------------------------------------------------------------------------- models

pub static MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "qwen3.8-27b",
        name: "Qwen3.8-27B",
        engine: "qwen35",
        summary: "Dense 27B model. Fast and light; the easiest one to start with.",
        weights: "about 17 GB with drafters, + the 2 GiB 4-bit prompt file (+26 GiB for int8 prompt weights)",
        drafter: "MTP head + DFlash2, picked by depth",
        min_free_gib: 8,
        license: "Apache-2.0",
        options: Q27_OPTIONS,
        reference: &[("numerics", OFF)],
        reference_note: "Turns off flash prompt attention and 4-bit prompt math: bit-for-bit reference output, slower prompts.",
        fast: &[("w4_fast", ON)],
        fast_note: "Turns on 4-bit prompt math in every layer: about 30% faster prompts, but it failed our quality gate (needs the 8 GiB .fast.w4i file).",
    },
    ModelSpec {
        id: "qwen3.8-flash-next",
        name: "Qwen3.8-Flash-Next",
        engine: "qwen4exp",
        summary: "Mixture-of-experts model with 256K context.",
        weights: "about 88 GB, streamed through an expert cache",
        drafter: "MTP head + repeat-text drafts",
        min_free_gib: 8,
        license: "Qwen Community License 1.0 (commercial services need a license from Qwen)",
        options: FN_OPTIONS,
        reference: &[("numerics", OFF)],
        reference_note: "Turns off the rounding-level speed-ups: exact reference output, slower.",
        fast: &[("hc16", ON)],
        fast_note: "Turns on 16-bit HC streams: faster prompts, but it failed our quality gate (the top token changes far more often).",
    },
    ModelSpec {
        id: "glm-5.3-flash",
        name: "GLM-5.3-Flash",
        engine: "glm",
        summary: "The largest model (MLA attention, KDA layers). Needs an otherwise idle machine.",
        weights: "about 110 GB",
        drafter: "built-in NextN layer",
        min_free_gib: 10,
        license: "MIT",
        options: GLM_OPTIONS,
        reference: &[("numerics", OFF)],
        reference_note: "Turns off fp16 prompt experts, 4096-row chunks and two-pass sparse attention: the earlier path.",
        fast: &[],
        fast_note: "No quality-reducing opt-in is recommended for GLM-5.3-Flash. Fast is the same as Balanced.",
    },
    ModelSpec {
        id: "mimo-v2.6-flash",
        name: "MiMo-V2.6-Flash",
        engine: "mimo",
        summary: "Mixture-of-experts model with 192K context and a DFlash drafter.",
        weights: "about 92 GB with the drafter",
        drafter: "DFlash (Q8_0)",
        min_free_gib: 11,
        license: "MIT",
        options: MIMO_OPTIONS,
        reference: &[("numerics", OFF)],
        reference_note: "Turns off wave32 expert mat-vecs and WMMA prompt attention: exact reference output.",
        fast: &[("iq1m", ON)],
        fast_note: "Turns on the low-bit expert prompt kernel: faster prompts, output quality may drop slightly.",
    },
    ModelSpec {
        id: "laguna-s-2.1",
        name: "Laguna-S-2.1",
        engine: "laguna",
        summary: "Mixture-of-experts coding model with sliding-window layers; 256K context fits.",
        weights: "about 68 GB",
        drafter: "prompt lookup + DFlash drafter (when its file is present)",
        min_free_gib: 8,
        license: "OpenMDW-1.1 (and poolside's Acceptable Use Policy)",
        options: LAG_OPTIONS,
        reference: &[("sparse", OFF), ("served_speed", OFF)],
        reference_note: "Full attention everywhere and the earlier serving path: exact reference output, slower.",
        fast: &[("verify_gemm", ON)],
        fast_note: "Turns on batched verify: much faster edits, but it failed our quality gate (outputs change; edits lost doc comments).",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    const NO_FAST_SUFFIX: &str = "Fast is the same as Balanced.";

    #[test]
    fn no_fast_notes_say_so() {
        for m in MODELS.iter().filter(|m| m.fast.is_empty()) {
            assert!(m.fast_note.ends_with(NO_FAST_SUFFIX), "{}", m.id);
        }
    }
}
