/*
 * ushuffle.c - re-entrant k-let preserving sequence shuffling.
 *
 * Algorithm (Euler walk + Wilson's random arborescence) by Minghui Jiang,
 * James Anderson, Joel Gillespie and Martin Mayne, BMC Bioinformatics 2008;
 * 9:192. Original BSD-3 notice in LICENSE.original. See ushuffle.h for what
 * this rewrite changes and why.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "prng.h"
#include "ushuffle.h"

#define USH_VERSION_STRING "2.0.0"

/* ------------------------------------------------------------------ */
/* data structures                                                      */
/* ------------------------------------------------------------------ */

/*
 * One graph vertex = one distinct (k-1)-let. `i_offset` indexes ctx->indices
 * instead of holding a pointer: 20 bytes instead of 32, and the whole block
 * stays relocatable when a buffer is grown.
 */
typedef struct {
    int32_t i_offset;   /* start of this vertex's out-edges in ctx->indices */
    int32_t n_indices;  /* out-degree                                       */
    int32_t i_indices;  /* cursor used while walking                        */
    int32_t i_sequence; /* position of the representative (k-1)-let         */
    int32_t next;       /* edge chosen by Wilson's algorithm                */
} ush_vertex;

/* Transient chained hash table entry, one per (k-1)-let position. */
typedef struct {
    int32_t next;       /* next entry in the bucket, or -1                  */
    int32_t i_sequence; /* position of the representative occurrence        */
    int32_t i_vertices; /* vertex id this position maps to                  */
} ush_hentry;

struct ush_ctx {
    ush_rng rng;

    char *seq;
    int32_t l;
    int k;
    int prepared;

    ush_vertex *vertices;
    uint8_t *intree;
    int32_t n_vertices;

    int32_t *indices;
    int32_t n_indices;

    int32_t root;

    size_t cap_seq;      /* all capacities are in elements, not bytes       */
    size_t cap_vertices;
    size_t cap_intree;
    size_t cap_indices;
};

/* ------------------------------------------------------------------ */
/* small helpers                                                        */
/* ------------------------------------------------------------------ */

/*
 * Grow *buf to at least `need` elements of `elem` bytes, remembering the
 * capacity so that repeated prepare() calls on one context stop reallocating.
 * Contents are not preserved -- every caller overwrites them immediately.
 */
static int ush_ensure(void **buf, size_t *cap, size_t need, size_t elem) {
    void *fresh;

    if (need == 0)
        return USH_OK;
    if (*cap >= need)
        return USH_OK;
    if (need > (size_t)-1 / elem)
        return USH_ERR_ALLOC;

    fresh = realloc(*buf, need * elem);
    if (fresh == NULL)
        return USH_ERR_ALLOC;

    *buf = fresh;
    *cap = need;
    return USH_OK;
}

/* FNV-1a over the (k-1)-let starting at `i`. */
static uint32_t ush_klet_hash(const char *s, int32_t i, int k) {
    uint32_t h = UINT32_C(2166136261);
    int j;

    for (j = 0; j < k - 1; j++) {
        h ^= (unsigned char)s[i + j];
        h *= UINT32_C(16777619);
    }
    return h;
}

static void ush_permute_chars(ush_rng *rng, char *t, int32_t l) {
    int32_t i, j;
    char tmp;

    for (i = l - 1; i > 0; i--) {
        j = (int32_t)ush_rng_below(rng, (uint64_t)i + 1);
        tmp = t[i];
        t[i] = t[j];
        t[j] = tmp;
    }
}

static void ush_permute_int32(ush_rng *rng, int32_t *t, int32_t l) {
    int32_t i, j, tmp;

    for (i = l - 1; i > 0; i--) {
        j = (int32_t)ush_rng_below(rng, (uint64_t)i + 1);
        tmp = t[i];
        t[i] = t[j];
        t[j] = tmp;
    }
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                            */
/* ------------------------------------------------------------------ */

ush_ctx *ush_ctx_new(void) {
    ush_ctx *ctx = (ush_ctx *)calloc(1, sizeof(ush_ctx));
    uint64_t entropy;

    if (ctx == NULL)
        return NULL;

    /*
     * Default seed. Nothing here needs to be cryptographic; it only has to
     * differ between processes and between contexts created in one process,
     * so that an unseeded user does not get the same "random" shuffle twice.
     */
    entropy = (uint64_t)(uintptr_t)ctx;
    entropy ^= (uint64_t)time(NULL) * UINT64_C(0x9E3779B97F4A7C15);
    entropy ^= (uint64_t)clock();
    ush_rng_seed(&ctx->rng, entropy, 0);

    return ctx;
}

void ush_ctx_free(ush_ctx *ctx) {
    if (ctx == NULL)
        return;
    free(ctx->seq);
    free(ctx->vertices);
    free(ctx->intree);
    free(ctx->indices);
    free(ctx);
}

void ush_ctx_seed(ush_ctx *ctx, uint64_t seed, uint64_t stream) {
    if (ctx != NULL)
        ush_rng_seed(&ctx->rng, seed, stream);
}

int32_t ush_ctx_length(const ush_ctx *ctx) {
    return ctx != NULL ? ctx->l : 0;
}

int ush_ctx_k(const ush_ctx *ctx) {
    return ctx != NULL ? ctx->k : 0;
}

int32_t ush_ctx_n_vertices(const ush_ctx *ctx) {
    return ctx != NULL ? ctx->n_vertices : 0;
}

size_t ush_memory_estimate(int32_t l, int k) {
    size_t n_lets;

    if (l <= 0 || k <= 1 || k >= l)
        return (size_t)(l > 0 ? l : 0);

    n_lets = (size_t)(l - k + 2);
    return (size_t)l                          /* sequence copy             */
           + n_lets * sizeof(ush_hentry)      /* hash entries (transient)  */
           + n_lets * sizeof(int32_t)         /* hash buckets (transient)  */
           + n_lets * sizeof(ush_vertex)      /* vertices (upper bound)    */
           + n_lets                           /* intree flags              */
           + n_lets * sizeof(int32_t);        /* edge list                 */
}

const char *ush_strerror(int code) {
    switch (code) {
    case USH_OK:
        return "ok";
    case USH_ERR_ALLOC:
        return "out of memory";
    case USH_ERR_ARG:
        return "invalid argument";
    case USH_ERR_STATE:
        return "internal invariant violated";
    case USH_ERR_UNPREPARED:
        return "context has no prepared sequence";
    default:
        return "unknown error";
    }
}

const char *ush_version(void) {
    return USH_VERSION_STRING;
}

/* ------------------------------------------------------------------ */
/* prepare: build the graph                                             */
/* ------------------------------------------------------------------ */

static int ush_build_graph(ush_ctx *ctx, int32_t n_lets) {
    ush_hentry *entries = NULL;
    int32_t *buckets = NULL;
    const char *seq = ctx->seq;
    const int k = ctx->k;
    int32_t i, j;
    int rc = USH_OK;

    /* calloc rather than malloc: every field is written below, but zeroing is
     * cheap next to the rest of the pass and it keeps compilers from warning
     * about a read they cannot prove is preceded by a write. */
    entries = (ush_hentry *)calloc((size_t)n_lets, sizeof(ush_hentry));
    buckets = (int32_t *)malloc((size_t)n_lets * sizeof(int32_t));
    if (entries == NULL || buckets == NULL) {
        rc = USH_ERR_ALLOC;
        goto done;
    }
    for (i = 0; i < n_lets; i++)
        buckets[i] = -1;

    /* Collapse identical (k-1)-lets onto one vertex each.
     *
     * Vertex ids are handed out in order of first occurrence and never depend
     * on the hash function, so replacing the original's floating point hash
     * with FNV-1a cannot change which shuffles are reachable. */
    ctx->n_vertices = 0;
    for (i = 0; i < n_lets; i++) {
        uint32_t code = ush_klet_hash(seq, i, k) % (uint32_t)n_lets;
        int32_t e;
        int found = 0;

        entries[i].next = -1;
        for (e = buckets[code]; e >= 0; e = entries[e].next) {
            if (memcmp(seq + entries[e].i_sequence, seq + i, (size_t)(k - 1)) == 0) {
                entries[i].i_sequence = entries[e].i_sequence;
                entries[i].i_vertices = entries[e].i_vertices;
                found = 1;
                break;
            }
        }
        if (!found) {
            entries[i].i_sequence = i;
            entries[i].i_vertices = ctx->n_vertices++;
            entries[i].next = buckets[code];
            buckets[code] = i;
        }
    }

    ctx->root = entries[n_lets - 1].i_vertices; /* the last let */

    rc = ush_ensure((void **)&ctx->vertices, &ctx->cap_vertices,
                    (size_t)ctx->n_vertices, sizeof(ush_vertex));
    if (rc != USH_OK)
        goto done;
    rc = ush_ensure((void **)&ctx->intree, &ctx->cap_intree,
                    (size_t)ctx->n_vertices, sizeof(uint8_t));
    if (rc != USH_OK)
        goto done;
    memset(ctx->vertices, 0, (size_t)ctx->n_vertices * sizeof(ush_vertex));

    /* out-degree and representative position per vertex */
    for (i = 0; i < n_lets; i++) {
        ush_vertex *v = &ctx->vertices[entries[i].i_vertices];

        v->i_sequence = entries[i].i_sequence;
        if (i < n_lets - 1) /* every let but the last contributes one edge */
            v->n_indices++;
    }

    ctx->n_indices = n_lets - 1;
    rc = ush_ensure((void **)&ctx->indices, &ctx->cap_indices,
                    (size_t)ctx->n_indices, sizeof(int32_t));
    if (rc != USH_OK)
        goto done;

    j = 0;
    for (i = 0; i < ctx->n_vertices; i++) {
        ctx->vertices[i].i_offset = j;
        j += ctx->vertices[i].n_indices;
        ctx->vertices[i].i_indices = 0;
    }

    for (i = 0; i < n_lets - 1; i++) {
        ush_vertex *u = &ctx->vertices[entries[i].i_vertices];

        ctx->indices[u->i_offset + u->i_indices++] = entries[i + 1].i_vertices;
    }

    /*
     * Invariant: only the root may be a sink. Wilson's walk divides by the
     * out-degree, so a non-root sink would mean an out-of-bounds read. The
     * structure of a Eulerian path guarantees this; checking it costs one
     * pass and turns a hypothetical memory error into a clean return code.
     */
    for (i = 0; i < ctx->n_vertices; i++) {
        if (ctx->vertices[i].n_indices == 0 && i != ctx->root) {
            rc = USH_ERR_STATE;
            goto done;
        }
    }

done:
    free(entries);
    free(buckets);
    return rc;
}

int ush_prepare(ush_ctx *ctx, const char *s, int32_t l, int k) {
    int rc;

    if (ctx == NULL || l < 0 || k < 1 || (s == NULL && l > 0))
        return USH_ERR_ARG;

    rc = ush_ensure((void **)&ctx->seq, &ctx->cap_seq, (size_t)l, sizeof(char));
    if (rc != USH_OK)
        return rc;
    if (l > 0)
        memcpy(ctx->seq, s, (size_t)l);

    ctx->l = l;
    ctx->k = k;
    ctx->n_vertices = 0;
    ctx->n_indices = 0;
    ctx->root = 0;
    ctx->prepared = 1;

    /* k >= l: the only shuffle is the input. k == 1: a plain permutation.
     * Both are handled in ush_generate() and need no graph. */
    if (k >= l || k <= 1)
        return USH_OK;

    rc = ush_build_graph(ctx, l - k + 2);
    if (rc != USH_OK)
        ctx->prepared = 0;
    return rc;
}

/* ------------------------------------------------------------------ */
/* generate: random Eulerian walk                                       */
/* ------------------------------------------------------------------ */

int ush_generate(ush_ctx *ctx, char *t) {
    ush_vertex *vertices;
    int32_t *indices;
    int32_t i, u, v;
    int32_t pos;
    const int k = ctx != NULL ? ctx->k : 0;

    if (ctx == NULL)
        return USH_ERR_ARG;
    if (!ctx->prepared)
        return USH_ERR_UNPREPARED;
    if (t == NULL && ctx->l > 0)
        return USH_ERR_ARG;
    if (ctx->l == 0)
        return USH_OK;

    if (k >= ctx->l) { /* exact copy */
        memcpy(t, ctx->seq, (size_t)ctx->l);
        return USH_OK;
    }
    if (k <= 1) { /* plain permutation */
        memcpy(t, ctx->seq, (size_t)ctx->l);
        ush_permute_chars(&ctx->rng, t, ctx->l);
        return USH_OK;
    }

    vertices = ctx->vertices;
    indices = ctx->indices;

    /* Wilson's algorithm: a uniformly random arborescence rooted at `root`.
     * This is what makes the resulting Eulerian walk uniform over all
     * sequences with the same k-let counts. */
    memset(ctx->intree, 0, (size_t)ctx->n_vertices);
    ctx->intree[ctx->root] = 1;
    for (i = 0; i < ctx->n_vertices; i++) {
        u = i;
        while (!ctx->intree[u]) {
            vertices[u].next =
                (int32_t)ush_rng_below(&ctx->rng, (uint64_t)vertices[u].n_indices);
            u = indices[vertices[u].i_offset + vertices[u].next];
        }
        u = i;
        while (!ctx->intree[u]) {
            ctx->intree[u] = 1;
            u = indices[vertices[u].i_offset + vertices[u].next];
        }
    }

    /* Order each vertex's out-edges: the arborescence edge goes last, the
     * rest are permuted uniformly. */
    for (i = 0; i < ctx->n_vertices; i++) {
        ush_vertex *vx = &vertices[i];
        int32_t *edge = indices + vx->i_offset;

        if (i != ctx->root && vx->n_indices > 0) {
            int32_t tmp = edge[vx->n_indices - 1];
            edge[vx->n_indices - 1] = edge[vx->next];
            edge[vx->next] = tmp;
            ush_permute_int32(&ctx->rng, edge, vx->n_indices - 1);
        } else {
            ush_permute_int32(&ctx->rng, edge, vx->n_indices);
        }
        vx->i_indices = 0;
    }

    /* Walk. The first (k-1)-let is fixed, every edge appends one character. */
    memcpy(t, ctx->seq, (size_t)(k - 1));
    pos = k - 1;
    u = 0;
    while (vertices[u].i_indices < vertices[u].n_indices) {
        if (pos >= ctx->l)
            return USH_ERR_STATE;
        v = indices[vertices[u].i_offset + vertices[u].i_indices];
        t[pos++] = ctx->seq[vertices[v].i_sequence + k - 2];
        vertices[u].i_indices++;
        u = v;
    }
    if (pos != ctx->l)
        return USH_ERR_STATE; /* the walk must consume every edge */

    return USH_OK;
}

int ush_shuffle(ush_ctx *ctx, const char *s, char *t, int32_t l, int k) {
    int rc = ush_prepare(ctx, s, l, k);

    if (rc != USH_OK)
        return rc;
    return ush_generate(ctx, t);
}
