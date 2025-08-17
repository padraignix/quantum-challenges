// ML-DSA-65 brute-force harness focused on "repeating pattern" seeds.
// Adds quick passes:
//   - Constant-byte seeds (v in 0..255)
//   - Periodic base-block seeds with period P in {2,4,8,16,24,32}
//       * t=1 hot position in the block (value 0..255)
//       * t=2 hot positions in the block (values 0..255 each)
//
// Build (static link recommended; add -lm if needed):
//   cc -O3 -march=native -std=c11 -I ./liboqs/build/include ml_dsa_bruteforce_periodic.c ./liboqs/build/lib/liboqs.a -o brute_periodic
// Run:
//   ./brute_periodic   (expects pk.bin in the current directory)
#include <oqs/oqs.h>
#include <oqs/rand.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEEDSPAN_MAX 256

static uint8_t seedbuf[SEEDSPAN_MAX];
static size_t  seedidx = 0;

static uint8_t *pk = NULL;
static uint8_t *sk = NULL;

static void custom_randombytes(uint8_t *out, size_t out_len) {
    for (size_t i = 0; i < out_len; i++) {
        out[i] = seedbuf[seedidx % SEEDSPAN_MAX];  // wrap
        seedidx++;
    }
}

static int load_file(const char *path, uint8_t **buf, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *b = (uint8_t *)malloc((size_t)sz);
    if (!b) { fclose(f); return -1; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return -1; }
    fclose(f);
    *buf = b;
    *len = (size_t)sz;
    return 0;
}

// Confirm our custom RNG is being used.
static int rng_uses_custom(void) {
    uint8_t probe[64];
    for (size_t i = 0; i < SEEDSPAN_MAX; i++) seedbuf[i] = (uint8_t)(i ^ 0xA5);
    seedidx = 0;
    OQS_randombytes_custom_algorithm(custom_randombytes);
    OQS_randombytes(probe, sizeof(probe));
    for (size_t i = 0; i < sizeof(probe); i++) {
        uint8_t exp = (i < SEEDSPAN_MAX) ? (uint8_t)(i ^ 0xA5) : 0x00;
        if (probe[i] != exp) return 0;
    }
    return 1;
}

// Measure number of RNG bytes consumed initially by keygen.
static size_t measure_consumed_first(OQS_SIG *sig) {
    for (size_t i = 0; i < SEEDSPAN_MAX; i++) seedbuf[i] = (uint8_t)(0xD0 ^ (i * 13));
    seedidx = 0;
    uint8_t *tpk = (uint8_t *)malloc(sig->length_public_key);
    uint8_t *tsk = (uint8_t *)malloc(sig->length_secret_key);
    if (!tpk || !tsk) { free(tpk); free(tsk); return 0; }
    (void)OQS_SIG_keypair(sig, tpk, tsk);
    free(tpk); free(tsk);
    return seedidx;
}

static inline int try_seed_and_check(OQS_SIG *sig,
                                     const uint8_t *target_pk, size_t target_pk_len,
                                     const uint8_t *seed, size_t seed_len,
                                     uint8_t *out_sk) {
    // Load seed into seedbuf and reset index
    memset(seedbuf, 0x00, SEEDSPAN_MAX);
    if (seed_len > SEEDSPAN_MAX) seed_len = SEEDSPAN_MAX;
    memcpy(seedbuf, seed, seed_len);
    seedidx = 0;

    if (OQS_SIG_keypair(sig, pk, sk) != OQS_SUCCESS) return 0;
    if (memcmp(pk, target_pk, target_pk_len) == 0) {
        memcpy(out_sk, sk, sig->length_secret_key);
        return 1;
    }
    return 0;
}

static void print_progress(size_t done, size_t total, clock_t start) {
    if (done == 0 || total == 0) return;
    if ((done & ((size_t)4096 - 1)) != 0) return; // every 4096
    double secs = (double)(clock() - start) / CLOCKS_PER_SEC;
    double rate = done / (secs > 0 ? secs : 1e-6);
    fprintf(stderr, "Tried %zu / %zu (%.2f%%) — %.0f/s\n",
            done, total, 100.0 * (double)done / (double)total, rate);
}

// Expand a periodic base block into the first N bytes of seed.
static void build_periodic_seed(uint8_t *dst, size_t N,
                                const uint8_t *block, size_t P) {
    for (size_t i = 0; i < N; i++) {
        dst[i] = block[i % P];
    }
}

int main(void) {
    // Load target pk
    uint8_t *target_pk = NULL; size_t target_len = 0;
    if (load_file("pk.bin", &target_pk, &target_len) != 0) {
        fprintf(stderr, "pk.bin not found.\n");
        return 1;
    }

    // Prepare ML-DSA-65
    OQS_SIG *sig = OQS_SIG_new("ML-DSA-65");
    if (!sig) {
        fprintf(stderr, "OQS_SIG_new(ML-DSA-65) failed; build liboqs with ML-DSA enabled.\n");
        free(target_pk);
        return 1;
    }
    if (target_len != sig->length_public_key) {
        fprintf(stderr, "Target pk length %zu != ML-DSA-65 pk length %zu\n",
                target_len, (size_t)sig->length_public_key);
        OQS_SIG_free(sig); free(target_pk);
        return 1;
    }

    // Buffers (allocate once)
    pk = (uint8_t *)malloc(sig->length_public_key);
    sk = (uint8_t *)malloc(sig->length_secret_key);
    uint8_t *found_sk = (uint8_t *)malloc(sig->length_secret_key);
    if (!pk || !sk || !found_sk) {
        fprintf(stderr, "alloc failed\n");
        OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); free(found_sk);
        return 1;
    }

    // Register custom RNG
    if (!rng_uses_custom()) {
        fprintf(stderr, "Custom RNG callback not engaged; liboqs version mismatch.\n");
        OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); free(found_sk);
        return 2;
    }

    // Determine initial RNG consumption (N)
    size_t N = SEEDSPAN_MAX;//measure_consumed_first(sig);
    //if (N == 0 || N > 128) N = 32; // reasonable fallback
    printf("Keygen consumed_first bytes: %zu\n", N);

    uint8_t seed[SEEDSPAN_MAX];
    memset(seed, 0x00, sizeof(seed));

    // Quick pass 0: constant-byte seeds (all bytes == v)
    {
        for (int v = 0; v < 256; v++) {
            memset(seed, (uint8_t)v, N);
            if (try_seed_and_check(sig, target_pk, target_len, seed, N, found_sk)) {
                printf("FOUND (constant-byte): v=%d\n", v);
                printf("Seed: ");
                for (size_t i = 0; i < N; i++) printf("%02x", seed[i]);
                printf("\n");
                goto SIGN_OUT;
            }
        }
        fprintf(stderr, "Constant-byte pass: no match.\n");
    }

    // Quick pass 1: periodic, t=1 (one hot position in block), P in set
    {
        size_t periods[] = {2, 4, 8, 16, 24, 32};
        size_t np = sizeof(periods) / sizeof(periods[0]);

        for (size_t pi = 0; pi < np; pi++) {
            size_t P = periods[pi];
            uint8_t block[32];
            if (P > sizeof(block)) continue;

            for (size_t pos = 0; pos < P; pos++) {
                for (int v = 0; v < 256; v++) {
                    memset(block, 0x00, P);
                    block[pos] = (uint8_t)v;
                    build_periodic_seed(seed, N, block, P);
                    if (try_seed_and_check(sig, target_pk, target_len, seed, N, found_sk)) {
                        printf("FOUND (periodic t=1): P=%zu pos=%zu val=%d\n", P, pos, v);
                        goto SIGN_OUT;
                    }
                }
            }
        }
        fprintf(stderr, "Periodic t=1 pass: no match.\n");
    }

    // Pass 2: periodic, t=2 (two hot positions), start with P in {4,8,16} for speed
    {
        size_t periods[] = {4, 8, 16};
        size_t np = sizeof(periods) / sizeof(periods[0]);

        for (size_t pi = 0; pi < np; pi++) {
            size_t P = periods[pi];
            uint8_t block[32];
            if (P > sizeof(block)) continue;

            // Count total tries for progress
            size_t pairs = (P >= 2) ? (P * (P - 1)) / 2 : 0;
            size_t total = pairs * 256ull * 256ull;
            size_t done = 0;
            clock_t clk0 = clock();

            for (size_t p1 = 0; p1 < P; p1++) {
                for (size_t p2 = p1 + 1; p2 < P; p2++) {
                    for (int v1 = 0; v1 < 256; v1++) {
                        for (int v2 = 0; v2 < 256; v2++) {
                            memset(block, 0x00, P);
                            block[p1] = (uint8_t)v1;
                            block[p2] = (uint8_t)v2;
                            build_periodic_seed(seed, N, block, P);
                            if (try_seed_and_check(sig, target_pk, target_len, seed, N, found_sk)) {
                                printf("FOUND (periodic t=2): P=%zu p1=%zu v1=%d, p2=%zu v2=%d\n",
                                       P, p1, v1, p2, v2);
                                goto SIGN_OUT;
                            }
                            print_progress(++done, total, clk0);
                        }
                    }
                }
            }
            fprintf(stderr, "Periodic t=2 (P=%zu): no match.\n", P);
        }
    }

    // Optional heavier pass: widen t=2 to P={24,32}
    // Uncomment to try if needed (can take a while at ~5–10k/s).
    /*
    {
        size_t periods[] = {24, 32};
        size_t np = sizeof(periods) / sizeof(periods[0]);
        for (size_t pi = 0; pi < np; pi++) {
            size_t P = periods[pi];
            uint8_t block[32];
            if (P > sizeof(block)) continue;
            size_t pairs = (P >= 2) ? (P * (P - 1)) / 2 : 0;
            size_t total = pairs * 256ull * 256ull;
            size_t done = 0;
            clock_t clk0 = clock();

            for (size_t p1 = 0; p1 < P; p1++) {
                for (size_t p2 = p1 + 1; p2 < P; p2++) {
                    for (int v1 = 0; v1 < 256; v1++) {
                        for (int v2 = 0; v2 < 256; v2++) {
                            memset(block, 0x00, P);
                            block[p1] = (uint8_t)v1;
                            block[p2] = (uint8_t)v2;
                            build_periodic_seed(seed, N, block, P);
                            if (try_seed_and_check(sig, target_pk, target_len, seed, N, found_sk)) {
                                printf("FOUND (periodic t=2): P=%zu p1=%zu v1=%d, p2=%zu v2=%d\n",
                                       P, p1, v1, p2, v2);
                                goto SIGN_OUT;
                            }
                            print_progress(++done, total, clk0);
                        }
                    }
                }
            }
            fprintf(stderr, "Periodic t=2 (P=%zu): no match.\n", P);
        }
    }
    */

    fprintf(stderr, "No match in periodic/constant passes. Next options: widen t=2 to P=24/32, or try t=3 for a subset.\n");
    OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); free(found_sk);
    return 3;

SIGN_OUT: ;
    // Sign fixed message and print first 8 bytes
    {


        
        const char *msg = "QuantumVillageChallenge2025";
        size_t mlen = strlen(msg);
        size_t siglen = sig->length_signature;
        uint8_t *signature = (uint8_t *)malloc(siglen);
        if (!signature) {
            fprintf(stderr, "alloc signature failed\n");
            OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); free(found_sk);
            return 1;
        }
        memset(seedbuf, 0x01, SEEDSPAN_MAX);
        seedidx = 0;
        if (OQS_SIG_sign(sig, signature, &siglen, (const uint8_t *)msg, mlen, found_sk) != OQS_SUCCESS) {
            fprintf(stderr, "sign failed\n");
            OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); free(found_sk); free(signature);
            return 1;
        }
        printf("Signature length: %zu\n", siglen);
        printf("Flag (first 8 bytes, pair-swapped): ");
        for (int i = 0; i < 8; i += 2) {
            printf("%02x%02x", signature[i+1], signature[i]);  // swap adjacent bytes
        }
        printf("\n");
        free(signature);
    }

    OQS_SIG_free(sig);
    free(target_pk);
    free(pk); free(sk); free(found_sk);
    return 0;
}