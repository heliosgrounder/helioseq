/*
 * ushuffle.h - re-entrant k-let preserving sequence shuffling.
 *
 * This is a reworked version of uShuffle by Minghui Jiang, James Anderson,
 * Joel Gillespie and Martin Mayne (BSD-3, see LICENSE.original). The Euler /
 * Wilson algorithm is theirs and is reproduced faithfully; the API around it
 * is new.
 *
 * What changed and why
 * --------------------
 *  * All state lives in a caller-owned `ush_ctx`. The original kept the
 *    sequence, its length, k and the whole graph in file-scope statics, so two
 *    live shufflers silently clobbered each other -- which shows up as heap
 *    corruption, not as a wrong answer (guma44/ushuffle#7).
 *  * The context owns a copy of the sequence. The original stored the caller's
 *    pointer and dereferenced it later, which made lifetime the caller's
 *    problem and produced dangling reads from the Python binding.
 *  * The random source is a per-context xoshiro256++ (see prng.h) instead of a
 *    global pointer to rand()/random(), so seeding is per object, reproducible
 *    across platforms, unbiased, and usable from several threads at once.
 *  * Allocation failure returns USH_ERR_ALLOC instead of calling exit(1).
 *  * Graph nodes address each other with int32 offsets rather than pointers,
 *    cutting peak memory from ~60 to ~42 bytes per input base.
 *
 * Threading: a context is not internally synchronised. Give each thread its
 * own context and the library is fully thread-safe; there is no shared state.
 */

#ifndef USHUFFLE_H
#define USHUFFLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ush_ctx ush_ctx;

enum {
    USH_OK = 0,
    USH_ERR_ALLOC = -1, /* out of memory                                   */
    USH_ERR_ARG = -2,   /* bad argument (null pointer, l < 0, k < 1, ...)   */
    USH_ERR_STATE = -3, /* internal invariant violated -- please report it  */
    USH_ERR_UNPREPARED = -4 /* ush_generate() before a successful prepare   */
};

/* ---- lifecycle ---------------------------------------------------- */

/* Returns NULL on allocation failure. */
ush_ctx *ush_ctx_new(void);
void ush_ctx_free(ush_ctx *ctx);

/*
 * Reseed the context's generator. `stream` selects an independent sequence for
 * the same seed; pass the worker index when fanning out over threads.
 * A fresh context is seeded from a non-deterministic source.
 */
void ush_ctx_seed(ush_ctx *ctx, uint64_t seed, uint64_t stream);

/* ---- shuffling ---------------------------------------------------- */

/*
 * Analyse `s` (length `l`, k-let size `k`) and build the de Bruijn style graph
 * once. The sequence is copied into the context, so `s` need not outlive the
 * call. Safe to call repeatedly on one context; buffers are reused.
 */
int ush_prepare(ush_ctx *ctx, const char *s, int32_t l, int k);

/*
 * Emit one shuffle of the prepared sequence into `t`, which must have room for
 * ush_ctx_length(ctx) bytes. No terminating NUL is written. Call as often as
 * you like; each call draws a fresh random Eulerian walk.
 */
int ush_generate(ush_ctx *ctx, char *t);

/* ush_prepare() followed by ush_generate(). */
int ush_shuffle(ush_ctx *ctx, const char *s, char *t, int32_t l, int k);

/* ---- introspection ------------------------------------------------ */

int32_t ush_ctx_length(const ush_ctx *ctx);
int ush_ctx_k(const ush_ctx *ctx);

/* Number of distinct (k-1)-lets, i.e. graph vertices. 0 before prepare. */
int32_t ush_ctx_n_vertices(const ush_ctx *ctx);

/*
 * Peak heap bytes ush_prepare() needs for a sequence of length `l` with k-let
 * size `k`. Exposed so callers can refuse a job instead of being OOM-killed:
 * a 250 Mbp chromosome still costs tens of gigabytes.
 */
size_t ush_memory_estimate(int32_t l, int k);

const char *ush_strerror(int code);
const char *ush_version(void);

#ifdef __cplusplus
}
#endif

#endif /* USHUFFLE_H */
