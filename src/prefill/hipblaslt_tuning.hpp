#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace strata::prefill::hipblaslt {

enum class InputType : uint8_t { f16, bf16 };

struct TuningRow {
    InputType type = InputType::bf16;
    int n = 0;
    int k = 0;
    int ldy = 0;
    int t_bucket = 0;
    int solution_id = -1;
};

class TuningTable {
public:
    bool load(const std::string& path, const std::string& expected_arch, int expected_version, std::string& err) {
        std::ifstream input(path);
        if (!input) {
            err = "cannot open tuning file";
            return false;
        }

        bool saw_header = false;
        std::string line;
        size_t line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == '#') continue;

            std::istringstream row(line.substr(first));
            if (!saw_header) {
                std::string magic;
                std::string arch;
                int version = 0;
                std::string extra;
                if (!(row >> magic >> arch >> version) || (row >> extra) || magic != "STRATA_HIPBLASLT_TUNING_V1") {
                    err = "invalid tuning header at line " + std::to_string(line_number);
                    return false;
                }
                // A table's solution IDs are scoped to its hipBLASLt version and its architecture FAMILY:
                // gfx1200/gfx1201 (RDNA4) share them, as gfx1100/gfx1101/gfx1102 (RDNA3) do.  A card can report
                // its sibling - an R9700 (gfx1201) under HSA_OVERRIDE_GFX_VERSION=12.0.0 reports gfx1200 - and
                // must still load its own calibration.  A solution ID that does not exist on the running card
                // fails the per-solution gate in prefill/gemm.cu (getAlgosFromIndex / matmulIsAlgoSupported) and
                // falls back to hipBLASEx, so the header gate does not need to be exact.
                const auto is_rdna4 = [](const std::string& a) { return a == "gfx1200" || a == "gfx1201"; };
                const auto is_rdna3 = [](const std::string& a) {
                    return a == "gfx1100" || a == "gfx1101" || a == "gfx1102";
                };
                if (arch != expected_arch &&
                    !(is_rdna4(arch) && is_rdna4(expected_arch)) &&
                    !(is_rdna3(arch) && is_rdna3(expected_arch))) {
                    err = "tuning architecture mismatch: file=" + arch + " runtime=" + expected_arch;
                    return false;
                }
                if (version != expected_version) {
                    err = "hipBLASLt version mismatch: file=" + std::to_string(version) +
                          " runtime=" + std::to_string(expected_version);
                    return false;
                }
                arch_ = std::move(arch);
                version_ = version;
                saw_header = true;
                continue;
            }

            std::string type_name;
            int n = 0, k = 0, ldy = 0, t_bucket = 0, solution_id = -1;
            std::string extra;
            if (!(row >> type_name >> n >> k >> ldy >> t_bucket >> solution_id) || (row >> extra) ||
                (type_name != "f16" && type_name != "bf16") || n <= 0 || k <= 0 || ldy < n || t_bucket <= 0 ||
                solution_id < 0) {
                err = "invalid tuning row at line " + std::to_string(line_number);
                return false;
            }
            TuningRow parsed;
            parsed.type = type_name == "f16" ? InputType::f16 : InputType::bf16;
            parsed.n = n;
            parsed.k = k;
            parsed.ldy = ldy;
            parsed.t_bucket = t_bucket;
            parsed.solution_id = solution_id;
            for (const auto& existing : rows_) {
                if (existing.type == parsed.type && existing.n == parsed.n && existing.k == parsed.k &&
                    existing.ldy == parsed.ldy && existing.t_bucket == parsed.t_bucket) {
                    err = "duplicate tuning key at line " + std::to_string(line_number);
                    return false;
                }
            }
            rows_.push_back(parsed);
        }
        if (!saw_header) {
            err = "tuning file has no header";
            return false;
        }
        return true;
    }

    const TuningRow* closest(InputType type, int n, int k, int ldy, int t) const {
        const TuningRow* best = nullptr;
        int64_t best_distance = std::numeric_limits<int64_t>::max();
        for (const auto& row : rows_) {
            if (row.type != type || row.n != n || row.k != k || row.ldy != ldy) continue;
            const int64_t distance = row.t_bucket >= t ? int64_t(row.t_bucket) - t : int64_t(t) - row.t_bucket;
            if (distance < best_distance ||
                (distance == best_distance && (!best || row.t_bucket < best->t_bucket))) {
                best = &row;
                best_distance = distance;
            }
        }
        return best;
    }

    const std::vector<TuningRow>& rows() const { return rows_; }
    const std::string& arch() const { return arch_; }
    int version() const { return version_; }

private:
    std::string arch_;
    int version_ = 0;
    std::vector<TuningRow> rows_;
};

// ---------------------------------------------------------------------------------------------------------------
// Where the tuning table comes from (talus#50).  One resolver for both prefill loaders (prefill/gemm.cu, the C++
// Flash-Next path, and tk_prefill_ctx_init in talus_kernels_prefill.cu, the Rust engines), so the rules cannot
// drift apart.
//
//   STRATA_HIPBLASLT_TUNING=<file>  that file; when it does not load (another hipBLASLt version, missing), the
//                                   runtime's sibling <dir of file>/<arch>-hipblaslt-<runtime version>.txt
//                                   (the pre-talus#50 behaviour, unchanged).
//   STRATA_HIPBLASLT_TUNING=<dir>   <dir>/<arch>-hipblaslt-<runtime version>.txt
//   STRATA_HIPBLASLT_TUNING= (empty), off, none or 0
//                                   no table (hipBLASEx), on purpose: the A/B switch.
//   unset                           auto-discovery, when the caller allows it: the first
//                                   <arch>-hipblaslt-<runtime version>.txt that loads in
//                                     <libdir>/hip                  release tarball (bin/hip) and flat deploys
//                                     <libdir>/../share/talus/hip   a packaged / FHS-style install
//                                     <libdir>/../tools/hip         a source-tree build (build/ next to tools/)
//                                   where <libdir> is the directory of the module (libtalus_kernels.so) holding
//                                   the caller.  Nothing found on an architecture with upstream tables -> the
//                                   caller prints one loud WARN (the prompt runs untuned).
// ---------------------------------------------------------------------------------------------------------------

/// `<arch>-hipblaslt-<version>.txt`
inline std::string tuning_file_name(const std::string& arch, int version) {
    return arch + "-hipblaslt-" + std::to_string(version) + ".txt";
}

/// Architectures for which tools/hip carries calibrated tables: running one of these without a table is a
/// misinstall worth a warning, not a choice.
inline bool arch_has_upstream_tables(const std::string& arch) {
    return arch == "gfx1100" || arch == "gfx1150" || arch == "gfx1151" || arch == "gfx1200" || arch == "gfx1201";
}

/// The auto-discovery directories for a module in `libdir`, in search order (empty `libdir`: none).
inline std::vector<std::string> default_tuning_dirs(const std::string& libdir) {
    if (libdir.empty()) return {};
    std::string d = libdir;
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    return {d + "/hip", d + "/../share/talus/hip", d + "/../tools/hip"};
}

/// The directory of the shared object (or executable) that contains `addr`; empty when unknown (Windows: the
/// auto-discovery is not wired there, STRATA_HIPBLASLT_TUNING still works).
inline std::string module_dir_of(const void* addr) {
#if !defined(_WIN32)
    Dl_info info{};
    if (addr != nullptr && dladdr(addr, &info) != 0 && info.dli_fname != nullptr && info.dli_fname[0] != '\0') {
        std::error_code ec;
        std::filesystem::path p(info.dli_fname);
        if (p.is_relative()) p = std::filesystem::absolute(p, ec);
        return p.parent_path().string();
    }
#else
    (void) addr;
#endif
    return {};
}

/// True for an STRATA_HIPBLASLT_TUNING value that turns the table off on purpose.
inline bool tuning_env_disables(const char* v) {
    if (v == nullptr) return false;
    const std::string s(v);
    return s.empty() || s == "0" || s == "off" || s == "OFF" || s == "none" || s == "NONE";
}

/// What the resolver did (for the log line and the tests).
struct TuningResolution {
    enum class Source { none, disabled, env_file, env_sibling, env_dir, auto_dir };
    Source source = Source::none;
    bool loaded = false;
    std::string path;                   ///< the table loaded
    std::string requested;              ///< the env value (env_*), else empty
    std::string first_error;            ///< env_sibling: why the named file did not load (the old log line)
    std::string error;                  ///< not loaded: why
    std::vector<std::string> searched;  ///< auto_dir / none: the candidates looked at
};

/// Find and load the table for (arch, version).  `env` is STRATA_HIPBLASLT_TUNING's value (nullptr = unset);
/// `auto_search` allows the unset case to search default_tuning_dirs(libdir).  `arch` has no feature suffix.
inline TuningResolution resolve_tuning_table(TuningTable& out, const char* env, bool auto_search,
                                             const std::string& libdir, const std::string& arch, int version) {
    TuningResolution r;
    if (env != nullptr) {
        if (tuning_env_disables(env)) {
            r.source = TuningResolution::Source::disabled;
            r.error = "disabled by STRATA_HIPBLASLT_TUNING";
            return r;
        }
        const std::string path(env);
        r.requested = path;
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec)) {
            std::string dir = path;
            while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
            const std::string candidate = dir + "/" + tuning_file_name(arch, version);
            r.source = TuningResolution::Source::env_dir;
            r.searched.push_back(candidate);
            TuningTable table;
            std::string err;
            if (table.load(candidate, arch, version, err)) {
                out = std::move(table);
                r.loaded = true;
                r.path = candidate;
            } else {
                r.error = err + " (" + candidate + ")";
            }
            return r;
        }
        // A file: exactly the pre-talus#50 rule (the named file, then the runtime's sibling next to it).
        r.source = TuningResolution::Source::env_file;
        TuningTable table;
        std::string error;
        if (table.load(path, arch, version, error)) {
            out = std::move(table);
            r.loaded = true;
            r.path = path;
            return r;
        }
        std::string dir(path);
        const auto slash = dir.find_last_of('/');
        dir = slash == std::string::npos ? std::string() : dir.substr(0, slash + 1);
        const std::string sibling = dir + tuning_file_name(arch, version);
        TuningTable sibling_table;
        std::string error2;
        if (sibling != path && sibling_table.load(sibling, arch, version, error2)) {
            out = std::move(sibling_table);
            r.source = TuningResolution::Source::env_sibling;
            r.loaded = true;
            r.path = sibling;
            r.first_error = error;
            return r;
        }
        r.error = error;
        return r;
    }
    if (!auto_search) {
        r.error = "STRATA_HIPBLASLT_TUNING is not set";
        return r;
    }
    for (const auto& dir : default_tuning_dirs(libdir)) {
        const std::string candidate = dir + "/" + tuning_file_name(arch, version);
        std::error_code ec;
        if (!std::filesystem::is_regular_file(candidate, ec)) {
            r.searched.push_back(candidate);
            continue;
        }
        TuningTable table;
        std::string err;
        if (table.load(candidate, arch, version, err)) {
            out = std::move(table);
            r.source = TuningResolution::Source::auto_dir;
            r.loaded = true;
            r.path = std::filesystem::path(candidate).lexically_normal().string();
            return r;
        }
        r.searched.push_back(candidate + " (" + err + ")");
    }
    r.error = "no " + tuning_file_name(arch, version) + " found";
    return r;
}

/// The one-line advice for "no table" on an architecture with upstream tables.
inline std::string untuned_warning(const std::string& arch, int version, const TuningResolution& r) {
    std::string where;
    for (const auto& s : r.searched) where += (where.empty() ? "" : ", ") + s;
    std::string msg = "talus: WARN: no hipBLASLt tuning table for " + arch + " / hipBLASLt " + std::to_string(version) +
                      " (" + (r.requested.empty() ? std::string() : "STRATA_HIPBLASLT_TUNING=" + r.requested + ": ") +
                      r.error + (where.empty() ? std::string() : "; looked for " + where) +
                      "): the prompt runs UNTUNED on hipBLASEx (slower prefill, different numerics). Fix: reinstall "
                      "Talus (the tables ship next to libtalus_kernels.so in hip/) or set STRATA_HIPBLASLT_TUNING to "
                      "a table file or to the directory holding " + tuning_file_name(arch, version) + ".";
    return msg;
}

}  // namespace strata::prefill::hipblaslt
