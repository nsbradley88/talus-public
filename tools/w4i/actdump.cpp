// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
// actdump (tools/w4i, from rc-a4 / w4cal): llama.cpp CPU forward over calibration sequences; dumps the input activations of the trunk
// projections (one file per layer x input: gdn_in, gdn_out, attn_in, attn_out, ffn_in, ffn_down) as fp16 rows [T][k].
// W4C_LAYERS (comma list of layer indices, default all), W4C_KINDS (comma list of kinds to dump, e.g. ffn_in,ffn_down; default all).
// Usage: actdump --model GGUF --out DIR SEQ.ids... (each file: comma/space separated token ids)
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct Cap {
    std::string out;
    std::map<std::string, FILE*> files;
    long long overflow = 0, rows = 0;
};
const char* kind_of(const char* w) {
    if (!std::strcmp(w, "attn_qkv")) return "gdn_in";
    if (!std::strcmp(w, "ssm_out")) return "gdn_out";
    if (!std::strcmp(w, "attn_q")) return "attn_in";
    if (!std::strcmp(w, "attn_output")) return "attn_out";
    if (!std::strcmp(w, "ffn_gate")) return "ffn_in";
    if (!std::strcmp(w, "ffn_down")) return "ffn_down";
    return nullptr;
}
bool parse(const ggml_tensor* t, int* L, const char** kind) {
    if (t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1]) return false;
    const char* n = t->src[0]->name;   // "blk.L.NAME.weight"
    if (std::strncmp(n, "blk.", 4) != 0) return false;
    char nm[64];
    int l;
    if (std::sscanf(n, "blk.%d.%63[^.]", &l, nm) != 2) return false;
    const char* k = kind_of(nm);
    if (!k) return false;
    *L = l;
    *kind = k;
    return true;
}
bool on_tensor(struct ggml_tensor* t, bool ask, void* user) {
    auto* c = (Cap*) user;
    int L;
    const char* kind;
    if (!parse(t, &L, &kind)) return ask ? false : true;
    static const std::string only = [] { const char* e = std::getenv("W4C_LAYERS"); return e ? "," + std::string(e) + "," : std::string(); }();
    if (!only.empty() && only.find("," + std::to_string(L) + ",") == std::string::npos) return ask ? false : true;
    static const std::string kinds = [] { const char* e = std::getenv("W4C_KINDS"); return e ? "," + std::string(e) + "," : std::string(); }();
    if (!kinds.empty() && kinds.find("," + std::string(kind) + ",") == std::string::npos) return ask ? false : true;
    if (ask) return true;
    const ggml_tensor* x = t->src[1];
    if (x->type != GGML_TYPE_F32) { std::fprintf(stderr, "src1 of %s not f32\n", t->name); return true; }
    const int64_t k = x->ne[0], T = x->ne[1] * x->ne[2] * x->ne[3];
    std::vector<float> v((size_t) k * T);
    if (ggml_is_contiguous(x)) ggml_backend_tensor_get(x, v.data(), 0, v.size() * 4);
    else {
        for (int64_t r = 0; r < T; ++r) ggml_backend_tensor_get(x, v.data() + r * k, r * x->nb[1], k * 4);
    }
    std::vector<ggml_fp16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        float f = v[i];
        if (std::fabs(f) > 65000.f) { f = f > 0 ? 65000.f : -65000.f; ++c->overflow; }
        h[i] = ggml_fp32_to_fp16(f);
    }
    std::string key = std::to_string(L) + "." + kind;
    FILE*& f = c->files[key];
    if (!f) f = std::fopen((c->out + "/" + key + ".f16").c_str(), "wb");
    std::fwrite(h.data(), 2, h.size(), f);
    c->rows += T;
    return true;
}
std::vector<llama_token> read_ids(const std::string& p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    for (char& ch : s) if (ch == ',' || ch == '[' || ch == ']') ch = ' ';
    std::istringstream is(s);
    std::vector<llama_token> v;
    int t;
    while (is >> t) v.push_back(t);
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::string model, out;
    std::vector<std::string> seqs;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model") model = argv[++i];
        else if (a == "--out") out = argv[++i];
        else seqs.push_back(a);
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* mdl = llama_model_load_from_file(model.c_str(), mp);
    if (!mdl) return 1;
    Cap cap;
    cap.out = out;
    for (const auto& sp : seqs) {
        std::vector<llama_token> ids = read_ids(sp);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (int) ids.size() + 16;
        cp.n_batch = cp.n_ctx;
        cp.n_ubatch = 512;
        if (const char* th = std::getenv("DUMP_THREADS")) cp.n_threads = cp.n_threads_batch = std::atoi(th);
        cp.op_offload = false;
        cp.cb_eval = on_tensor;
        cp.cb_eval_user_data = &cap;
        llama_context* ctx = llama_init_from_model(mdl, cp);
        llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); ++i) {   // logits for every row: no last-layer row pruning
            b.token[i] = ids[i]; b.pos[i] = (llama_pos) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (ids.size() <= 4096 || i % 8 == 7 || i + 1 == ids.size()) ? 1 : 0;
        }
        b.n_tokens = (int) ids.size();
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
        llama_batch_free(b);
        llama_free(ctx);
        std::fprintf(stderr, "%s: %zu tokens, overflow so far %lld\n", sp.c_str(), ids.size(), cap.overflow);
    }
    for (auto& [k, f] : cap.files) std::fclose(f);
    llama_model_free(mdl);
    llama_backend_free();
    return 0;
}
