//! Prompt cut & message checkpoint boundary planning for Flash-Next (`qwen4exp`),
//! ported 1:1 from:
//! - `include/strata/program/message_boundary.hpp`
//! - `src/program/generate.cpp` (`turn_at`, `root_at`, `message_at`, `cuts` computation).

/// 1:1 port of `strata::program::message_checkpoint_boundary`
/// (`include/strata/program/message_boundary.hpp`).
pub fn message_checkpoint_boundary(
    ids: &[i64],
    resume: i64,
    turn_at: i64,
    turn_token: i64,
    min_prefix: i64,
    max_tail: i64,
) -> i64 {
    let n = ids.len() as i64;
    if resume < 0
        || turn_token < 0
        || turn_at <= resume
        || turn_at >= n
        || ids[turn_at as usize] != turn_token
        || min_prefix < 0
        || max_tail <= 0
    {
        return -1;
    }
    for i in ((resume + 1)..turn_at).rev() {
        if turn_at - i > max_tail {
            return -1;
        }
        if ids[i as usize] == turn_token {
            return if i >= min_prefix { i } else { -1 };
        }
    }
    -1
}

/// Output of `compute_prompt_cuts` for a `--serve` `GEN` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCutPlan {
    pub turn_at: i64,
    pub root_at: i64,
    pub message_at: i64,
    pub cuts: Vec<i64>,
}

/// 1:1 port of `src/program/generate.cpp` lines 11928–11960:
/// computes `turn_at`, `root_at`, `message_at`, and the sorted deduplicated `cuts` vector.
pub fn compute_prompt_cuts(
    ids: &[i64],
    resume: i64,
    read_from: i64,
    resumed_from0: bool,
    reread_to: i64,
    prompt_cache: i32,
    turn_token: i64,
    tail_role_token: i64,
    prompt_cache_root: i64,
    req_ckpt: bool,
    multi_gpu: bool,
    message_checkpoint_env: bool,
    pin_at: i64,
) -> PromptCutPlan {
    let n = ids.len() as i64;

    // 1. turn_at
    let mut turn_at = -1i64;
    if prompt_cache > 0 && turn_token >= 0 {
        for i in ((resume + 1)..n).rev() {
            if ids[i as usize] == turn_token {
                turn_at = i;
                break;
            }
        }
    }
    // #458 (opt-in): a short turn of --tail-role-token's role right before the new assistant turn
    if turn_at > 0 && tail_role_token >= 0 {
        for i in ((resume + 1)..turn_at).rev() {
            if ids[i as usize] == turn_token {
                if (i + 1) < n && ids[(i + 1) as usize] == tail_role_token {
                    turn_at = i;
                }
                break;
            }
        }
    }
    let last_turn = turn_at;
    if !req_ckpt {
        turn_at = -1;
    }

    // 2. root_at
    let mut root_at = -1i64;
    if prompt_cache > 0
        && turn_token >= 0
        && prompt_cache_root > 0
        && (read_from == 0 || resumed_from0)
    {
        for i in 1..last_turn {
            if ids[i as usize] == turn_token {
                if i >= prompt_cache_root {
                    root_at = i;
                }
                break;
            }
        }
    }

    // 3. message_at
    let message_at = if message_checkpoint_env && !multi_gpu && prompt_cache > 0 {
        message_checkpoint_boundary(ids, resume, turn_at, turn_token, 8192, 1024)
    } else {
        -1
    };

    // 4. cuts: sorted deduplicated positions in (read_from, n - 1]
    let mut cuts = Vec::with_capacity(6);
    for c in [reread_to, pin_at, root_at, message_at, turn_at, n - 1] {
        if c > read_from && c < n {
            cuts.push(c);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();

    PromptCutPlan {
        turn_at,
        root_at,
        message_at,
        cuts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_checkpoint_boundary_matches_cpp() {
        let im_start = 151644i64;
        let mut ids = vec![10i64; 10000];
        ids[0] = im_start;
        ids[2500] = im_start;
        ids[9000] = im_start;
        ids[9500] = im_start;

        // 9500 - 9000 = 500 <= 1024 and 9000 >= 8192 -> returns 9000
        let b = message_checkpoint_boundary(&ids, 0, 9500, im_start, 8192, 1024);
        assert_eq!(b, 9000);

        // If min_prefix is 9200, 9000 < 9200 -> returns -1
        let b2 = message_checkpoint_boundary(&ids, 0, 9500, im_start, 9200, 1024);
        assert_eq!(b2, -1);

        // If max_tail is 400, 9500 - 9000 = 500 > 400 -> returns -1
        let b3 = message_checkpoint_boundary(&ids, 0, 9500, im_start, 8192, 400);
        assert_eq!(b3, -1);
    }

    #[test]
    fn compute_prompt_cuts_orders_and_deduplicates() {
        let im_start = 151644i64;
        let user_tok = 872i64;
        let asst_tok = 77091i64;
        let mut ids = vec![42i64; 10000];
        ids[0] = im_start;
        ids[2500] = im_start;
        ids[2501] = user_tok;
        ids[9000] = im_start;
        ids[9001] = user_tok;
        ids[9500] = im_start;
        ids[9501] = asst_tok;

        let plan = compute_prompt_cuts(
            &ids, 0,     // resume
            0,     // read_from
            false, // resumed_from0
            -1,    // reread_to
            8,     // prompt_cache
            im_start, -1,    // tail_role_token
            2048,  // prompt_cache_root
            true,  // req_ckpt
            false, // multi_gpu
            true,  // message_checkpoint_env
            -1,    // pin_at
        );
        assert_eq!(plan.turn_at, 9500);
        assert_eq!(plan.root_at, 2500);
        assert_eq!(plan.message_at, 9000);
        assert_eq!(plan.cuts, vec![2500, 9000, 9500, 9999]);
    }
}
