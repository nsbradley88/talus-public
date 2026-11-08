#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program {

inline int64_t message_checkpoint_boundary(const std::vector<int64_t>& ids, int64_t resume,
        int64_t turn_at, int64_t turn_token, int64_t min_prefix = 8192, int64_t max_tail = 1024) {
    if (resume < 0 || turn_token < 0 || turn_at <= resume || turn_at >= int64_t(ids.size()) ||
        ids[std::size_t(turn_at)] != turn_token || min_prefix < 0 || max_tail <= 0) return -1;
    for (int64_t i = turn_at - 1; i > resume; --i) {
        if (turn_at - i > max_tail) return -1;
        if (ids[std::size_t(i)] == turn_token) return i >= min_prefix ? i : -1;
    }
    return -1;
}

// talus#60: a checkpoint at the prompt's end (n - 1, the state the reply starts from) for a request whose cuts leave
// none near it.  A hybrid-recurrent model (Flash-Next: GDN layers) resumes only from the live state when it is a strict
// prefix of the next prompt, or from a checkpoint; checkpoints were taken only at turn tokens (and every
// --prompt-cache-every tokens), so a raw /completion prompt without a turn token kept none: the same prompt again, a
// longer one, or a /slots restore of it reused 0 tokens.  Returns n - 1, or -1 when a checkpoint of this request is
// within --short-read tokens of the end already (a chat prompt: its turn checkpoint) or there is no conversation cache.
// `resumed_checkpoint`: the checkpoint this request mounted (0 = none, or it resumed from the live state, which is not
// kept): it counts as near too.  A chat prompt sent again resumes at its turn checkpoint with no turn left to read, and
// an extra checkpoint at n - 1 would move where the next turn of that chat resumes, and so change that turn's output.
inline int64_t prompt_end_checkpoint(int64_t n, int64_t turn_at, int64_t root_at, int64_t message_at, int64_t pin_at,
        int prompt_cache, bool req_ckpt, int64_t short_read, int64_t resumed_checkpoint) {
    const int64_t end = n - 1;
    if (prompt_cache <= 0 || !req_ckpt || end < 1) return -1;
    int64_t nearest = 0;
    for (const int64_t c : {turn_at, root_at, message_at, pin_at, resumed_checkpoint})
        if (c <= end && c > nearest) nearest = c;
    return end - nearest > (short_read > 0 ? short_read : 0) ? end : -1;
}

} // namespace strata::program
