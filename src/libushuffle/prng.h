/*
 * prng.h - xoshiro256++ pseudo-random generator with unbiased bounded sampling.
 *
 * Written for ushuffle-ng. Based on the public-domain xoshiro256++ by
 * David Blackman and Sebastiano Vigna (https://prng.di.unimi.it/) and on
 * splitmix64 by Sebastiano Vigna.
 *
 * Why this exists: the original uShuffle used the platform's rand()/random()
 * through a global function pointer. On Windows RAND_MAX is 32767, so a
 * Fisher-Yates pass over a sequence longer than ~32 kb cannot reach indices
 * above that bound at all, silently breaking uniformity. `x % n` also
 * introduces modulo bias. Both problems matter for a tool whose entire
 * purpose is producing *uniformly* distributed shuffles.
 *
 * Everything here is inline, allocation-free and keeps its state in a caller
 * owned struct, so it is re-entrant and safe to use from several threads as
 * long as each thread owns its own ush_rng.
 */

#ifndef USHUFFLE_PRNG_H
#define USHUFFLE_PRNG_H

#include <stdint.h>

typedef struct {
    uint64_t s[4];
} ush_rng;

/* ------------------------------------------------------------------ */

static uint64_t ush_rng__splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static uint64_t ush_rng__rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/*
 * Seed the generator. `stream` lets callers derive independent streams from a
 * single user-visible seed (worker i of a thread pool uses stream = i), so a
 * parallel run stays reproducible without the workers sharing state.
 */
static void ush_rng_seed(ush_rng *r, uint64_t seed, uint64_t stream) {
    uint64_t state = seed ^ (stream * UINT64_C(0xD1B54A32D192ED03));
    int i;
    for (i = 0; i < 4; i++)
        r->s[i] = ush_rng__splitmix64(&state);
    if ((r->s[0] | r->s[1] | r->s[2] | r->s[3]) == 0)
        r->s[0] = UINT64_C(0x9E3779B97F4A7C15); /* the all-zero state is fixed */
}

static uint64_t ush_rng_next(ush_rng *r) {
    const uint64_t result = ush_rng__rotl(r->s[0] + r->s[3], 23) + r->s[0];
    const uint64_t t = r->s[1] << 17;

    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = ush_rng__rotl(r->s[3], 45);

    return result;
}

/*
 * Uniform integer in [0, n).
 *
 * Masked rejection sampling: draw only as many low bits as `n` needs and retry
 * on overflow. Exactly uniform (no modulo bias), needs no 128-bit arithmetic
 * (so it behaves identically on MSVC, 32-bit targets and everywhere else), and
 * the expected number of draws is below 2.
 */
static uint64_t ush_rng_below(ush_rng *r, uint64_t n) {
    uint64_t mask, v;

    if (n <= 1)
        return 0;

    mask = n - 1;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;

    do {
        v = ush_rng_next(r) & mask;
    } while (v >= n);

    return v;
}

#endif /* USHUFFLE_PRNG_H */
