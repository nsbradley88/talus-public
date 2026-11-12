// S10: the W32 rule table, one source: compiled into native_mmvq.cu (runtime lookup and the tsum twins' compile-time
// lookup) and parsed by the Rust host (talus_kernels::w32). First match wins; {type (-1 = any), first column count, last, R, W[, CG[, SG]]}.
// gfx1151 (Strix Halo): in-engine, UD-IQ4_XS verify windows at 2K (gfx1151 verify-window table)
// (STRATA_MMVQ_W32_TAB tuning table, ms per verify pass by type): 2..3 columns R 1 (R 2 for the IQ3 / IQ2
// grids), 4 columns R 2 at <= 8 waves (IQ4_XS R 1); 5..9 columns R 2 at <= 8 waves for every type (IQ4_XS 8 columns
// 35.9 ms against 42.4 at R 1, 33.0 at 1 column), Q4_K R 4.
    // Q4_K (KQ8) at 4..5 columns (MiMo DFlash verify) R 1 W 0, as at 2..3: MiMo 8192x4096 4 / 5 cols 132 / 132
    // -> 111 / 113 us, 4096x12288 156 / 163 -> 145 / 150, 4096x14848 186 / 192 -> 173 / 187 (mvb graph sweep with a
    // dependent kernel; every column bitwise the 1-column result, mvbx);
    // STRATA_MMVQ_W32_TAB="12:4-5:2:8" restores
    {12, 4, 5, 1, 0},
    // Graph sweep with a dependent kernel, 27B shapes 5120x17408 / 17408x5120):
    // IQ2_S / IQ2_XS at 3..5 columns R 4 W 8 (IQ2_S 5 cols 206 / 187 -> 187 / 179 us, IQ2_XS 4 cols 175 / 173 -> 167 /
    // 153), Q3_K at 3..6 R 4 W 8 (6 cols 375 / 354 -> 345 / 345) and at 8 R 4 W 0 (458 / 434 -> 431 / 409). Bitwise
    // (every instance's columns are the 1-column result, mb/mvbx.cpp); STRATA_MMVQ_W32_TAB overrides.
    {22, 3, 5, 4, 8}, {17, 3, 5, 4, 8}, {11, 3, 6, 4, 8}, {11, 8, 8, 4, 0},
    {21, 2, 3, 2, 0}, {18, 2, 3, 2, 0}, {22, 2, 3, 2, 0}, {17, 2, 3, 2, 0}, {11, 2, 3, 2, 0}, {-1, 2, 3, 1, 0},
    // IQ3_XXS at 4 columns R 2 at <= 8 waves (W 8), not the default occupancy: 27B 5120x17408 / 17408x5120
    // 240-250 -> 176-188 us (mvb graph sweep); STRATA_MMVQ_W32_TAB="18:4:2:0" restores
    {23, 4, 4, 1, 8}, {18, 4, 4, 2, 8}, {-1, 4, 4, 2, 8},
    {-1, 5, 10, 2, 8},   // (Q4_K / Q5_K: KQ8Traits R 2 selected on gfx1151)
    // 11..13 columns: R 2 in one pass at <= 6 waves (<= 256 VGPRs; 12-row window 117 ms against 128 in groups); 16:
    // that for IQ4_XS / Q4_K / IQ2_XS / Q2_K / IQ4_NL (IQ4_XS 53.7 ms against 60.7), else and at 14..15 R 2 in column
    // groups of 8 (16-row window 155 ms against 198 at R 1)
    {-1, 11, 13, 2, 6}, {23, 16, 16, 2, 6}, {12, 16, 16, 2, 6}, {13, 16, 16, 2, 6}, {17, 16, 16, 2, 6}, {10, 16, 16, 2, 6}, {20, 16, 16, 2, 6},
    {-1, 14, 16, 2, 8, 8},
