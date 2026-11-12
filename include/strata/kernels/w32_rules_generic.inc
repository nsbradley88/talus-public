// S10: the W32 rule table, one source: compiled into native_mmvq.cu (runtime lookup and the tsum twins' compile-time
// lookup) and parsed by the Rust host (talus_kernels::w32). First match wins; {type (-1 = any), first column count, last, R, W[, CG[, SG]]}.
    {10, 1, 1, 2, 0}, {17, 1, 1, 2, 0}, {22, 1, 1, 2, 0}, {11, 1, 1, 4, 0}, {-1, 1, 1, 1, 0},
    {12, 2, 5, 2, 0}, {13, 2, 5, 2, 0}, {14, 2, 5, 2, 0}, {21, 2, 5, 4, 0}, {11, 2, 5, 4, 0}, {10, 2, 5, 4, 0},
    {-1, 2, 5, 1, 0},
    {12, 6, 7, 2, 12}, {13, 6, 7, 2, 12}, {14, 6, 7, 2, 12}, {21, 6, 7, 4, 12}, {11, 6, 7, 4, 12}, {10, 6, 7, 4, 12},
    {-1, 6, 7, 1, 12},
    {12, 8, 8, 2, 8}, {13, 8, 8, 2, 8}, {14, 8, 8, 2, 8}, {21, 8, 8, 4, 8}, {11, 8, 8, 4, 8}, {10, 8, 8, 4, 8},
    {-1, 8, 8, 1, 8}, {-1, 9, 16, 1, 8},
