/*
 * C-level test harness for libushuffle.
 *
 * Covers what the Python suite cannot reach directly: memory safety under
 * ASan/UBSan, isolation between contexts (the bug that made the original
 * corrupt the heap), and real multi-threaded use. Build and run with
 * `make ctest` or `make ctest-asan` from the repository root.
 *
 * `--golden` prints fixed-seed shuffles as JSON; tests/data/golden.json is
 * generated from it so the pure-Python backend can be checked against the C
 * implementation byte for byte.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ushuffle.h"

#ifdef USH_TEST_THREADS
#include <pthread.h>
#endif

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        checks++;                                                              \
        if (!(cond)) {                                                         \
            failures++;                                                        \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
        }                                                                      \
    } while (0)

/* ------------------------------------------------------------------ */
/* helpers                                                              */
/* ------------------------------------------------------------------ */

/* Count k-lets by sorting the substrings; O(n log n) and independent of the
 * library's own hashing, so it is a real cross-check rather than a tautology. */
/* qsort gives the comparator no context, so the width lives in a thread-local:
 * a plain global would be a data race in the threaded test. */
#if defined(_MSC_VER)
#define USH_TLS __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define USH_TLS _Thread_local
#else
#define USH_TLS __thread
#endif

static USH_TLS int g_k;

static int cmp_klet_thunk(const void *a, const void *b) {
    return memcmp(*(const char **)a, *(const char **)b, (size_t)g_k);
}

static char *klet_signature(const char *s, int l, int k) {
    int n = l - k + 1;
    const char **ptrs;
    char *out;
    int i;

    if (n <= 0)
        return NULL;
    ptrs = (const char **)malloc((size_t)n * sizeof(char *));
    for (i = 0; i < n; i++)
        ptrs[i] = s + i;
    g_k = k;
    qsort(ptrs, (size_t)n, sizeof(char *), cmp_klet_thunk);

    out = (char *)malloc((size_t)n * (size_t)k + 1);
    for (i = 0; i < n; i++)
        memcpy(out + (size_t)i * (size_t)k, ptrs[i], (size_t)k);
    out[(size_t)n * (size_t)k] = '\0';
    free(ptrs);
    return out;
}

static int klets_equal(const char *a, const char *b, int l, int k) {
    char *sa = klet_signature(a, l, k);
    char *sb = klet_signature(b, l, k);
    int eq;

    if (sa == NULL || sb == NULL) {
        free(sa);
        free(sb);
        return memcmp(a, b, (size_t)l) == 0;
    }
    eq = strcmp(sa, sb) == 0;
    free(sa);
    free(sb);
    return eq;
}

static void random_seq(char *buf, int l, const char *alpha, unsigned *state) {
    int n = (int)strlen(alpha);
    int i;

    for (i = 0; i < l; i++) {
        *state = *state * 1103515245u + 12345u;
        buf[i] = alpha[(*state >> 16) % (unsigned)n];
    }
}

/* ------------------------------------------------------------------ */
/* tests                                                                */
/* ------------------------------------------------------------------ */

static void test_klets_preserved(void) {
    static const char *alphabets[] = {"AC", "ACGT", "ACDEFGHIKLMNPQRSTVWY"};
    unsigned state = 7u;
    int a, k, trial;

    for (a = 0; a < 3; a++) {
        for (k = 1; k <= 6; k++) {
            for (trial = 0; trial < 40; trial++) {
                int l = 1 + (int)(state % 200u);
                char *s = (char *)malloc((size_t)l);
                char *t = (char *)malloc((size_t)l);
                ush_ctx *ctx = ush_ctx_new();
                int rc;

                random_seq(s, l, alphabets[a], &state);
                ush_ctx_seed(ctx, (uint64_t)trial, (uint64_t)k);
                rc = ush_shuffle(ctx, s, t, l, k);
                CHECK(rc == USH_OK, "shuffle failed: %s", ush_strerror(rc));
                CHECK(klets_equal(s, t, l, k),
                      "k-let counts differ (alphabet=%s l=%d k=%d)", alphabets[a], l, k);
                /* k-let preservation implies j-let preservation for j < k */
                if (k > 1)
                    CHECK(klets_equal(s, t, l, k - 1),
                          "(k-1)-let counts differ (l=%d k=%d)", l, k);
                ush_ctx_free(ctx);
                free(s);
                free(t);
            }
        }
    }
}

static void test_prefix_and_suffix_fixed(void) {
    const char *s = "ACGTTGCAACGTAGCTAGCTTTACG";
    int l = (int)strlen(s);
    char t[64];
    ush_ctx *ctx = ush_ctx_new();
    int k, i;

    for (k = 2; k <= 4; k++) {
        ush_prepare(ctx, s, l, k);
        for (i = 0; i < 200; i++) {
            CHECK(ush_generate(ctx, t) == USH_OK, "generate failed");
            /* uShuffle keeps the first (k-1)-let; the last is then forced. */
            CHECK(memcmp(t, s, (size_t)(k - 1)) == 0, "prefix changed for k=%d", k);
            CHECK(memcmp(t + l - (k - 1), s + l - (k - 1), (size_t)(k - 1)) == 0,
                  "suffix changed for k=%d", k);
        }
    }
    ush_ctx_free(ctx);
}

/*
 * The regression test for guma44/ushuffle#7. With file-scope state, preparing
 * a second sequence invalidated the first shuffler and ush_generate wrote past
 * the caller's buffer. Two live contexts must be completely independent.
 */
static void test_context_isolation(void) {
    const char *a = "ACGTACGTAC";
    const char *b = "GGGGCCCCGGGGCCCCGGGGCCCCGGGG";
    int la = (int)strlen(a), lb = (int)strlen(b);
    char ta[16], tb[64];
    ush_ctx *ca = ush_ctx_new();
    ush_ctx *cb = ush_ctx_new();
    int i;

    CHECK(ush_prepare(ca, a, la, 2) == USH_OK, "prepare a");
    CHECK(ush_prepare(cb, b, lb, 2) == USH_OK, "prepare b");

    for (i = 0; i < 100; i++) {
        memset(ta, '#', sizeof(ta));
        memset(tb, '#', sizeof(tb));
        CHECK(ush_generate(ca, ta) == USH_OK, "generate a");
        CHECK(ush_generate(cb, tb) == USH_OK, "generate b");
        CHECK(klets_equal(a, ta, la, 2), "context a produced foreign content");
        CHECK(klets_equal(b, tb, lb, 2), "context b produced foreign content");
        CHECK(ta[la] == '#', "context a wrote past its buffer");
        CHECK(tb[lb] == '#', "context b wrote past its buffer");
    }
    ush_ctx_free(ca);
    ush_ctx_free(cb);
}

static void test_reproducibility(void) {
    const char *s = "ACGTTGCAACGTAGCTAGCTTTACGGCTATCGATCG";
    int l = (int)strlen(s);
    char t1[64], t2[64], t3[64];
    ush_ctx *c1 = ush_ctx_new();
    ush_ctx *c2 = ush_ctx_new();
    int i, differed = 0;

    ush_ctx_seed(c1, 42, 0);
    ush_ctx_seed(c2, 42, 0);
    ush_prepare(c1, s, l, 2);
    ush_prepare(c2, s, l, 2);
    for (i = 0; i < 50; i++) {
        ush_generate(c1, t1);
        ush_generate(c2, t2);
        CHECK(memcmp(t1, t2, (size_t)l) == 0, "same seed diverged at draw %d", i);
    }

    /* Different streams must not track each other. */
    ush_ctx_seed(c2, 42, 1);
    ush_prepare(c2, s, l, 2);
    ush_ctx_seed(c1, 42, 0);
    ush_prepare(c1, s, l, 2);
    for (i = 0; i < 50; i++) {
        ush_generate(c1, t1);
        ush_generate(c2, t3);
        if (memcmp(t1, t3, (size_t)l) != 0)
            differed = 1;
    }
    CHECK(differed, "streams 0 and 1 produced identical output 50 times");

    ush_ctx_free(c1);
    ush_ctx_free(c2);
}

static void test_edge_cases(void) {
    ush_ctx *ctx = ush_ctx_new();
    char t[16];
    int rc;

    CHECK(ush_prepare(ctx, "", 0, 2) == USH_OK, "empty sequence rejected");
    CHECK(ush_generate(ctx, t) == USH_OK, "empty generate failed");

    CHECK(ush_prepare(ctx, "A", 1, 2) == USH_OK, "length-1 rejected");
    CHECK(ush_generate(ctx, t) == USH_OK, "length-1 generate failed");
    CHECK(t[0] == 'A', "length-1 content changed");

    /* k == l and k > l are exact copies */
    CHECK(ush_prepare(ctx, "ACGT", 4, 4) == USH_OK, "k == l rejected");
    CHECK(ush_generate(ctx, t) == USH_OK, "k == l generate failed");
    CHECK(memcmp(t, "ACGT", 4) == 0, "k == l is not an exact copy");

    CHECK(ush_prepare(ctx, "ACGT", 4, 9) == USH_OK, "k > l rejected");
    ush_generate(ctx, t);
    CHECK(memcmp(t, "ACGT", 4) == 0, "k > l is not an exact copy");

    /* k == 1 is a plain permutation, so composition but not order survives */
    CHECK(ush_prepare(ctx, "AAAACCCC", 8, 1) == USH_OK, "k == 1 rejected");
    ush_generate(ctx, t);
    CHECK(klets_equal("AAAACCCC", t, 8, 1), "k == 1 changed composition");

    /* embedded NUL bytes must not truncate anything */
    CHECK(ush_prepare(ctx, "AC\0GTAC\0GT", 10, 2) == USH_OK, "NUL rejected");
    CHECK(ush_generate(ctx, t) == USH_OK, "NUL generate failed");
    CHECK(klets_equal("AC\0GTAC\0GT", t, 10, 2), "NUL byte mishandled");

    /* bad arguments */
    CHECK(ush_prepare(ctx, "ACGT", 4, 0) == USH_ERR_ARG, "k == 0 accepted");
    CHECK(ush_prepare(ctx, "ACGT", -1, 2) == USH_ERR_ARG, "negative length accepted");
    CHECK(ush_prepare(NULL, "ACGT", 4, 2) == USH_ERR_ARG, "NULL context accepted");

    {
        ush_ctx *fresh = ush_ctx_new();
        rc = ush_generate(fresh, t);
        CHECK(rc == USH_ERR_UNPREPARED, "generate before prepare returned %d", rc);
        ush_ctx_free(fresh);
    }

    /* homopolymers have exactly one valid shuffle */
    CHECK(ush_prepare(ctx, "AAAAAAAA", 8, 2) == USH_OK, "homopolymer rejected");
    ush_generate(ctx, t);
    CHECK(memcmp(t, "AAAAAAAA", 8) == 0, "homopolymer changed");

    ush_ctx_free(ctx);
}

static void test_reuse(void) {
    ush_ctx *ctx = ush_ctx_new();
    char big[512], small[8], out[512];
    unsigned state = 99u;
    int i;

    /* Alternate long and short sequences on one context: buffers are reused,
     * so a stale length or capacity would show up here. */
    for (i = 0; i < 50; i++) {
        random_seq(big, 500, "ACGT", &state);
        CHECK(ush_shuffle(ctx, big, out, 500, 3) == USH_OK, "long shuffle failed");
        CHECK(klets_equal(big, out, 500, 3), "long shuffle broke k-lets");

        random_seq(small, 6, "ACGT", &state);
        CHECK(ush_shuffle(ctx, small, out, 6, 2) == USH_OK, "short shuffle failed");
        CHECK(klets_equal(small, out, 6, 2), "short shuffle broke k-lets");
    }
    ush_ctx_free(ctx);
}

/*
 * Uniformity. For a short sequence we can enumerate every string over the
 * observed alphabet, keep those with the same k-let counts and the same first
 * (k-1)-let -- that set is exactly the support of the shuffle -- and check the
 * observed counts against uniform with a chi-square statistic.
 */
/* Backtracking enumeration of the distinct permutations of s's letters that
 * also match its k-let counts and its first (k-1)-let. */
typedef struct {
    const char *s;
    int l, k;
    char letters[256];
    int n_letters;
    int remaining[256];
    char *buf;
    char **support;
    int n_support, cap;
} support_builder;

static void support_emit(support_builder *sb) {
    if (sb->n_support == sb->cap) {
        sb->cap = sb->cap ? sb->cap * 2 : 64;
        sb->support = (char **)realloc(sb->support, (size_t)sb->cap * sizeof(char *));
    }
    sb->support[sb->n_support] = (char *)malloc((size_t)sb->l + 1);
    memcpy(sb->support[sb->n_support], sb->buf, (size_t)sb->l);
    sb->support[sb->n_support][sb->l] = '\0';
    sb->n_support++;
}

static void support_recurse(support_builder *sb, int depth) {
    int i;

    if (depth == sb->l) {
        if (klets_equal(sb->s, sb->buf, sb->l, sb->k))
            support_emit(sb);
        return;
    }
    for (i = 0; i < sb->n_letters; i++) {
        if (sb->remaining[i] == 0)
            continue;
        /* the first (k-1) characters are fixed by the algorithm */
        if (depth < sb->k - 1 && sb->letters[i] != sb->s[depth])
            continue;
        sb->remaining[i]--;
        sb->buf[depth] = sb->letters[i];
        support_recurse(sb, depth + 1);
        sb->remaining[i]++;
    }
}

static void test_uniformity(void) {
    /* 12 bases, 180 distinct shuffles -- small enough to enumerate, large
     * enough for the chi-square to have real power. */
    const char *s = "AACAGATAACAG";
    const int l = 12, k = 2;
    const int n_draws = 120000;
    char t[16];
    support_builder sb;
    char **support;
    int n_support;
    long *observed;
    double chi2 = 0.0, expected;
    long total = 0;
    ush_ctx *ctx;
    int i, j;

    memset(&sb, 0, sizeof(sb));
    sb.s = s;
    sb.l = l;
    sb.k = k;
    sb.buf = (char *)malloc((size_t)l);
    for (i = 0; i < l; i++) {
        for (j = 0; j < sb.n_letters; j++)
            if (sb.letters[j] == s[i])
                break;
        if (j == sb.n_letters)
            sb.letters[sb.n_letters++] = s[i];
        sb.remaining[j]++;
    }
    support_recurse(&sb, 0);
    free(sb.buf);
    support = sb.support;
    n_support = sb.n_support;

    CHECK(n_support > 20, "support too small to test (%d)", n_support);
    if (n_support <= 1) {
        free(support);
        return;
    }

    /* sort the support so each draw can be located with a binary search */
    g_k = l;
    qsort(support, (size_t)n_support, sizeof(char *), cmp_klet_thunk);

    observed = (long *)calloc((size_t)n_support, sizeof(long));
    ctx = ush_ctx_new();
    ush_ctx_seed(ctx, 20260915, 0);
    ush_prepare(ctx, s, l, k);
    for (i = 0; i < n_draws; i++) {
        int lo = 0, hi = n_support - 1;

        ush_generate(ctx, t);
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            int c = memcmp(t, support[mid], (size_t)l);

            if (c == 0) {
                observed[mid]++;
                total++;
                break;
            }
            if (c < 0)
                hi = mid - 1;
            else
                lo = mid + 1;
        }
    }
    CHECK(total == n_draws, "%ld draws fell outside the support", (long)n_draws - total);

    expected = (double)n_draws / (double)n_support;
    for (j = 0; j < n_support; j++) {
        double d = (double)observed[j] - expected;
        chi2 += d * d / expected;
    }
    /*
     * df = n_support - 1; compare against mean + 6 sd = df + 6*sqrt(2*df).
     * Loose enough never to flake, tight enough to catch a skewed sampler
     * (the original's Windows rand() path fails this by a wide margin).
     */
    {
        double df = (double)(n_support - 1);
        double bound = df + 6.0 * sqrt(2.0 * df);

        printf("  uniformity: support=%d chi2=%.1f bound=%.1f\n", n_support, chi2, bound);
        CHECK(chi2 < bound, "chi2=%.1f exceeds %.1f for %d outcomes", chi2, bound,
              n_support);
    }

    for (j = 0; j < n_support; j++)
        free(support[j]);
    free(support);
    free(observed);
    ush_ctx_free(ctx);
}

#ifdef USH_TEST_THREADS
typedef struct {
    const char *seq;
    int l, k;
    int iterations;
    int ok;
} thread_arg;

static void *thread_body(void *p) {
    thread_arg *arg = (thread_arg *)p;
    ush_ctx *ctx = ush_ctx_new();
    char *t = (char *)malloc((size_t)arg->l);
    int i;

    arg->ok = 1;
    ush_ctx_seed(ctx, 1234, (uint64_t)(uintptr_t)p);
    ush_prepare(ctx, arg->seq, arg->l, arg->k);
    for (i = 0; i < arg->iterations; i++) {
        if (ush_generate(ctx, t) != USH_OK || !klets_equal(arg->seq, t, arg->l, arg->k)) {
            arg->ok = 0;
            break;
        }
    }
    free(t);
    ush_ctx_free(ctx);
    return NULL;
}

static void test_threads(void) {
    enum { N = 8 };
    pthread_t tid[N];
    thread_arg args[N];
    char seq[2000];
    unsigned state = 3u;
    int i;

    random_seq(seq, 2000, "ACGT", &state);
    for (i = 0; i < N; i++) {
        args[i].seq = seq;
        args[i].l = 2000;
        args[i].k = 2 + (i % 4);
        args[i].iterations = 200;
        args[i].ok = 0;
        pthread_create(&tid[i], NULL, thread_body, &args[i]);
    }
    for (i = 0; i < N; i++) {
        pthread_join(tid[i], NULL);
        CHECK(args[i].ok, "thread %d produced a bad shuffle", i);
    }
}
#endif

/* ------------------------------------------------------------------ */
/* golden vectors                                                       */
/* ------------------------------------------------------------------ */

static void emit_golden(void) {
    struct {
        const char *seq;
        int k;
        uint64_t seed;
        uint64_t stream;
        int n;
    } cases[] = {
        {"ACGTACGTAGCTAGCTAAGGCCTT", 2, 42, 0, 5},
        {"ACGTACGTAGCTAGCTAAGGCCTT", 3, 42, 0, 5},
        {"ACGTACGTAGCTAGCTAAGGCCTT", 1, 7, 3, 5},
        {"AAAACCCCGGGGTTTTACGTACGTTGCATGCA", 2, 2026, 1, 5},
        {"MKVLAAGIVGLNLGGKVAAMKVLAAG", 2, 99, 0, 5},
        {"GCGCGCGCGCGCGCGCGC", 4, 5, 0, 5},
    };
    size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    size_t c;

    printf("[\n");
    for (c = 0; c < n_cases; c++) {
        int l = (int)strlen(cases[c].seq);
        char *t = (char *)malloc((size_t)l + 1);
        ush_ctx *ctx = ush_ctx_new();
        int i;

        ush_ctx_seed(ctx, cases[c].seed, cases[c].stream);
        ush_prepare(ctx, cases[c].seq, l, cases[c].k);
        printf("  {\"seq\": \"%s\", \"k\": %d, \"seed\": %llu, \"stream\": %llu,\n",
               cases[c].seq, cases[c].k, (unsigned long long)cases[c].seed,
               (unsigned long long)cases[c].stream);
        printf("   \"draws\": [");
        for (i = 0; i < cases[c].n; i++) {
            ush_generate(ctx, t);
            t[l] = '\0';
            printf("%s\"%s\"", i ? ", " : "", t);
        }
        printf("]}%s\n", c + 1 < n_cases ? "," : "");
        ush_ctx_free(ctx);
        free(t);
    }
    printf("]\n");
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--golden") == 0) {
        emit_golden();
        return 0;
    }

    printf("libushuffle %s\n", ush_version());
    printf("  memory estimate for 1 Mbp, k=2: %.1f MB\n",
           (double)ush_memory_estimate(1000000, 2) / 1048576.0);

    test_klets_preserved();
    test_prefix_and_suffix_fixed();
    test_context_isolation();
    test_reproducibility();
    test_edge_cases();
    test_reuse();
    test_uniformity();
#ifdef USH_TEST_THREADS
    test_threads();
#endif

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
