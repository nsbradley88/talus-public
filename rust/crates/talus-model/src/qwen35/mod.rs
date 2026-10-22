//! Qwen3.5 / Qwen3.6 model & engine adapter (`qwen35`, `qwen35moe`).
//!
//! Bit-for-bit Rust port of:
//! - `include/strata/core/model_desc.hpp`
//! - `src/model/qwen35.cpp`
//! - `src/qwen35/q36_engine.cu`

pub mod engine;
pub mod ffi;
pub mod graphs;
pub mod w4i;
pub mod weights;

#[cfg(test)]
mod ssamp_tests;
#[cfg(test)]
mod tests;

pub use engine::*;
pub use ffi::{CoreError, Result};
pub use graphs::*;
pub use weights::*;

use ffi::{ensure, GgufFileExt};
use talus_core_api::{
    GraphBuilder, GraphKey, HiddenTaps, LogitRows, ModelAdapter, ModelDesc, SeqState, StateLayout,
    StreamHandle, WindowIo,
};
use talus_gguf::GgufFile;

/// Dense 27B hybrid 3:1 GDN:GQA architecture adapter (`qwen35`).
#[derive(Debug, Clone)]
pub struct Qwen35Arch {
    geom: ModelGeometry,
    desc: ModelDesc,
    layout: StateLayout,
}

impl Default for Qwen35Arch {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen35Arch {
    pub fn new() -> Self {
        let geom = reference_dense();
        let desc = geom.to_model_desc();
        let layout = geom.to_state_layout();
        Self { geom, desc, layout }
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let geom = describe_gguf(gguf)?;
        ensure!(!geom.is_moe(), "expected dense qwen35 GGUF, got MoE");
        let desc = geom.to_model_desc();
        let layout = geom.to_state_layout();
        Ok(Self { geom, desc, layout })
    }

    pub fn geometry(&self) -> &ModelGeometry {
        &self.geom
    }

    pub fn arch_name(&self) -> &str {
        &self.geom.arch
    }

    pub fn num_layers(&self) -> usize {
        self.geom.n_layer as usize
    }

    pub fn hidden_size(&self) -> usize {
        self.geom.n_embd as usize
    }

    pub fn vocab_size(&self) -> usize {
        self.geom.n_vocab as usize
    }
}

impl ModelAdapter for Qwen35Arch {
    fn desc(&self) -> &ModelDesc {
        &self.desc
    }

    fn state_layout(&self) -> StateLayout {
        self.layout.clone()
    }

    fn prefill(
        &mut self,
        _st: &mut SeqState,
        _ids: &[u32],
        _pos0: u32,
        _want: LogitRows,
        _s: &StreamHandle,
    ) -> Result<()> {
        Ok(())
    }

    fn window(&mut self, _st: &mut SeqState, _w: &WindowIo, _s: &StreamHandle) -> Result<()> {
        Ok(())
    }

    fn commit(&mut self, _st: &mut SeqState, _n_keep: u32, _s: &StreamHandle) -> Result<()> {
        Ok(())
    }

    fn hidden_taps(&self) -> HiddenTaps {
        HiddenTaps::default()
    }

    fn capture(&mut self, _key: GraphKey, _g: &mut GraphBuilder) -> Result<()> {
        Ok(())
    }
}

/// MoE 35B-A3B hybrid 3:1 GDN:GQA architecture adapter (`qwen35moe`).
#[derive(Debug, Clone)]
pub struct Qwen35MoeArch {
    geom: ModelGeometry,
    desc: ModelDesc,
    layout: StateLayout,
}

impl Default for Qwen35MoeArch {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen35MoeArch {
    pub fn new() -> Self {
        let geom = reference_moe();
        let desc = geom.to_model_desc();
        let layout = geom.to_state_layout();
        Self { geom, desc, layout }
    }

    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let geom = describe_gguf(gguf)?;
        ensure!(geom.is_moe(), "expected MoE qwen35moe GGUF, got dense");
        let desc = geom.to_model_desc();
        let layout = geom.to_state_layout();
        Ok(Self { geom, desc, layout })
    }

    pub fn geometry(&self) -> &ModelGeometry {
        &self.geom
    }

    pub fn arch_name(&self) -> &str {
        &self.geom.arch
    }

    pub fn num_layers(&self) -> usize {
        self.geom.n_layer as usize
    }

    pub fn hidden_size(&self) -> usize {
        self.geom.n_embd as usize
    }

    pub fn vocab_size(&self) -> usize {
        self.geom.n_vocab as usize
    }
}

impl ModelAdapter for Qwen35MoeArch {
    fn desc(&self) -> &ModelDesc {
        &self.desc
    }

    fn state_layout(&self) -> StateLayout {
        self.layout.clone()
    }

    fn prefill(
        &mut self,
        _st: &mut SeqState,
        _ids: &[u32],
        _pos0: u32,
        _want: LogitRows,
        _s: &StreamHandle,
    ) -> Result<()> {
        Ok(())
    }

    fn window(&mut self, _st: &mut SeqState, _w: &WindowIo, _s: &StreamHandle) -> Result<()> {
        Ok(())
    }

    fn commit(&mut self, _st: &mut SeqState, _n_keep: u32, _s: &StreamHandle) -> Result<()> {
        Ok(())
    }

    fn hidden_taps(&self) -> HiddenTaps {
        HiddenTaps::default()
    }

    fn capture(&mut self, _key: GraphKey, _g: &mut GraphBuilder) -> Result<()> {
        Ok(())
    }
}

/// Verify that all required Qwen3.5/3.6 tensors exist in `gguf` for `geom`.
pub fn check_required_tensors(gguf: &GgufFile, geom: &ModelGeometry) -> Result<()> {
    let require = |name: &str| -> Result<()> {
        ensure!(
            gguf.tensor(name).is_some(),
            "missing required tensor '{name}' in GGUF"
        );
        Ok(())
    };
    require("token_embd.weight")?;
    require("output_norm.weight")?;
    for i in 0..geom.n_layer {
        let p = format!("blk.{i}.");
        require(&format!("{p}attn_norm.weight"))?;
        require(&format!("{p}post_attention_norm.weight"))?;
        if geom.is_full_attention_layer(i as usize) {
            if gguf.tensor(&format!("{p}attn_qkv.weight")).is_none() {
                require(&format!("{p}attn_q.weight"))?;
                require(&format!("{p}attn_k.weight"))?;
                require(&format!("{p}attn_v.weight"))?;
            }
            require(&format!("{p}attn_output.weight"))?;
            require(&format!("{p}attn_q_norm.weight"))?;
            require(&format!("{p}attn_k_norm.weight"))?;
        } else {
            require(&format!("{p}attn_qkv.weight"))?;
            require(&format!("{p}attn_gate.weight"))?;
            require(&format!("{p}ssm_alpha.weight"))?;
            require(&format!("{p}ssm_beta.weight"))?;
            require(&format!("{p}ssm_a"))?;
            require(&format!("{p}ssm_dt.bias"))?;
            require(&format!("{p}ssm_conv1d.weight"))?;
            require(&format!("{p}ssm_norm.weight"))?;
            require(&format!("{p}ssm_out.weight"))?;
        }
        if geom.is_moe() {
            require(&format!("{p}ffn_gate_inp.weight"))?;
            require(&format!("{p}ffn_gate_exps.weight"))?;
            require(&format!("{p}ffn_up_exps.weight"))?;
            require(&format!("{p}ffn_down_exps.weight"))?;
            require(&format!("{p}ffn_gate_shexp.weight"))?;
            require(&format!("{p}ffn_up_shexp.weight"))?;
            require(&format!("{p}ffn_down_shexp.weight"))?;
            require(&format!("{p}ffn_gate_inp_shexp.weight"))?;
        } else {
            require(&format!("{p}ffn_gate.weight"))?;
            require(&format!("{p}ffn_up.weight"))?;
            require(&format!("{p}ffn_down.weight"))?;
        }
    }
    Ok(())
}
