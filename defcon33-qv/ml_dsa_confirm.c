// Confirm recovered ML-DSA-65 key, sign message, verify, and dump artifacts.
// Reconstructs the key using a constant-byte seed (v=1) for the first N bytes consumed by keygen.
//
// Build:
//   cc -O3 -march=native -std=c11 -I ./liboqs/build/include ml_dsa_confirm_and_dump.c ./liboqs/build/lib/liboqs.a -o confirm
//
// Usage examples:
//   ./confirm -m "exact challenge string here"
//   ./confirm -f challenge.txt
//
// Outputs (in current directory):
//   found_sk.bin      (secret key)
//   found_pk.bin      (public key generated)
//   signature.bin     (raw signature bytes)
//   signature.hex     (hex-encoded signature, single line)
//   signature.b64     (base64-encoded signature, single line)
//
// Notes:
// - pk.bin must exist (the target public key you extracted).
// - Message handling:
//     - -m signs the literal UTF-8 bytes of the string (no newline).
//     - -f signs the exact raw file bytes (no modification).
// - If both -m and -f are omitted, it defaults to the literal string "QuantumVillageChallenge2025".
#include <oqs/oqs.h>
#include <oqs/rand.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEEDSPAN_MAX 256

static uint8_t seedbuf[SEEDSPAN_MAX];
static size_t  seedidx = 0;

static void custom_randombytes(uint8_t *out, size_t out_len) {
    for (size_t i = 0; i < out_len; i++) {
        if (seedidx < SEEDSPAN_MAX) out[i] = seedbuf[seedidx++];
        else out[i] = 0x00;
    }
}

static int load_file(const char *path, uint8_t **buf, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    uint8_t *b = (uint8_t *)malloc((size_t)sz);
    if (!b) { fclose(f); return -1; }
    if (sz > 0 && fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return -1; }
    fclose(f);
    *buf = b;
    *len = (size_t)sz;
    return 0;
}

static int save_file(const char *path, const uint8_t *buf, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (len && fwrite(buf, 1, len, f) != len) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

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

static void b64_encode(const uint8_t *in, size_t inlen, char **out_str) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t outlen = ((inlen + 2) / 3) * 4;
    char *out = (char *)malloc(outlen + 1);
    if (!out) { *out_str = NULL; return; }
    size_t i = 0, j = 0;
    while (i + 2 < inlen) {
        uint32_t v = (in[i] << 16) | (in[i+1] << 8) | in[i+2];
        out[j++] = tbl[(v >> 18) & 0x3F];
        out[j++] = tbl[(v >> 12) & 0x3F];
        out[j++] = tbl[(v >> 6) & 0x3F];
        out[j++] = tbl[v & 0x3F];
        i += 3;
    }
    if (i < inlen) {
        uint32_t v = in[i] << 16;
        if (i + 1 < inlen) v |= in[i+1] << 8;
        out[j++] = tbl[(v >> 18) & 0x3F];
        out[j++] = tbl[(v >> 12) & 0x3F];
        if (i + 1 < inlen) {
            out[j++] = tbl[(v >> 6) & 0x3F];
            out[j++] = '=';
        } else {
            out[j++] = '=';
            out[j++] = '=';
        }
    }
    out[j] = '\0';
    *out_str = out;
}

static void print_hex_line(const uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; i++) printf("%02x", buf[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    const char *msg_str = "QuantumVillageChallenge2025";
    const char *msg_file = NULL;

    // Parse args (simple)
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "-m") == 0 && a + 1 < argc) { msg_str = argv[++a]; msg_file = NULL; }
        else if (strcmp(argv[a], "-f") == 0 && a + 1 < argc) { msg_file = argv[++a]; }
        else if (strcmp(argv[a], "-h") == 0 || strcmp(argv[a], "--help") == 0) {
            fprintf(stderr, "Usage: %s [-m \"message\"] [-f message_file]\n", argv[0]);
            return 0;
        }
    }

    // Load target pk
    uint8_t *target_pk = NULL; size_t target_len = 0;
    if (load_file("pk.bin", &target_pk, &target_len) != 0) {
        fprintf(stderr, "pk.bin not found in current directory.\n");
        return 1;
    }

    // Prepare ML-DSA-65
    OQS_SIG *sig = OQS_SIG_new("ML-DSA-65");
    if (!sig) {
        fprintf(stderr, "OQS_SIG_new(ML-DSA-65) failed; was liboqs built with ML-DSA enabled?\n");
        free(target_pk);
        return 1;
    }
    if (target_len != sig->length_public_key) {
        fprintf(stderr, "Target pk length %zu != ML-DSA-65 pk length %zu\n",
                target_len, (size_t)sig->length_public_key);
        OQS_SIG_free(sig); free(target_pk);
        return 1;
    }

    // Engage custom RNG
    if (!rng_uses_custom()) {
        fprintf(stderr, "Custom RNG callback not engaged; liboqs version mismatch.\n");
        OQS_SIG_free(sig); free(target_pk);
        return 2;
    }

    // Determine N = initial RNG bytes consumed by keygen
    size_t N = SEEDSPAN_MAX; //size_t N = measure_consumed_first(sig);
    //if (N == 0 || N > 128) N = 32;
    printf("Keygen consumed_first bytes: %zu\n", N);

    // Reconstruct with constant-byte seed v=1
    //uint8_t seed[SEEDSPAN_MAX]; memset(seed, 0x00, sizeof(seed));
    //memset(seed, 0x01, N);
    //memset(seedbuf, 0x00, sizeof(seedbuf));
    //memcpy(seedbuf, seed, N);
    //seedidx = 0;

    // recontructing with all 0x01 bytes
    memset(seedbuf, 0x01, SEEDSPAN_MAX);  // fill ENTIRE stream with 0x01
    seedidx = 0;

    // Generate keypair
    uint8_t *pk = (uint8_t *)malloc(sig->length_public_key);
    uint8_t *sk = (uint8_t *)malloc(sig->length_secret_key);
    if (!pk || !sk) {
        fprintf(stderr, "alloc failed\n");
        OQS_SIG_free(sig); free(target_pk); free(pk); free(sk);
        return 1;
    }
    if (OQS_SIG_keypair(sig, pk, sk) != OQS_SUCCESS) {
        fprintf(stderr, "keypair failed\n");
        OQS_SIG_free(sig); free(target_pk); free(pk); free(sk);
        return 1;
    }

    // Confirm pk matches
    if (memcmp(pk, target_pk, target_len) != 0) {
        fprintf(stderr, "Generated pk does not match target pk.bin (unexpected given earlier result).\n");
        OQS_SIG_free(sig); free(target_pk); free(pk); free(sk);
        return 3;
    }
    printf("Public key match: OK\n");

    // Save keys
    (void)save_file("found_pk.bin", pk, sig->length_public_key);
    (void)save_file("found_sk.bin", sk, sig->length_secret_key);

    // Load message
    uint8_t *msg = NULL; size_t mlen = 0;
    if (msg_file) {
        if (load_file(msg_file, &msg, &mlen) != 0) {
            fprintf(stderr, "Failed to read message file: %s\n", msg_file);
            OQS_SIG_free(sig); free(target_pk); free(pk); free(sk);
            return 1;
        }
        printf("Signing raw file bytes from: %s (len=%zu)\n", msg_file, mlen);
    } else {
        mlen = strlen(msg_str);
        msg = (uint8_t *)malloc(mlen);
        if (!msg) { fprintf(stderr, "alloc msg failed\n"); OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); return 1; }
        memcpy(msg, msg_str, mlen);
        printf("Signing literal string: \"%s\" (len=%zu)\n", msg_str, mlen);
    }

    // Sign
    size_t siglen = sig->length_signature;
    uint8_t *signature = (uint8_t *)malloc(siglen);
    if (!signature) {
        fprintf(stderr, "alloc signature failed\n");
        OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); free(msg);
        return 1;
    }
    if (OQS_SIG_sign(sig, signature, &siglen, msg, mlen, sk) != OQS_SUCCESS) {
        fprintf(stderr, "sign failed\n");
        OQS_SIG_free(sig); free(target_pk); free(pk); free(sk); free(msg); free(signature);
        return 1;
    }
    printf("Signature length: %zu\n", siglen);

    // Verify
    OQS_STATUS vrc = OQS_SIG_verify(sig, msg, mlen, signature, siglen, pk);
    printf("Verification: %s\n", vrc == OQS_SUCCESS ? "OK" : "FAIL");

    // Dump signature
    (void)save_file("signature.bin", signature, siglen);

    // Hex (one line)
    FILE *hex = fopen("signature.hex", "w");
    if (hex) {
        for (size_t i = 0; i < siglen; i++) fprintf(hex, "%02x", signature[i]);
        fprintf(hex, "\n");
        fclose(hex);
    }

    // Base64 (one line)
    char *b64 = NULL;
    b64_encode(signature, siglen, &b64);
    if (b64) {
        FILE *b64f = fopen("signature.b64", "w");
        if (b64f) { fprintf(b64f, "%s\n", b64); fclose(b64f); }
        free(b64);
    }

    // Print a quick prefix to screen
    printf("Signature first 8 bytes (hex): ");
    for (int i = 0; i < 8 && i < (int)siglen; i++) printf("%02x", signature[i]);
    printf("\n");

    // Clean up
    OQS_SIG_free(sig);
    free(target_pk);
    free(pk); free(sk);
    free(msg);
    free(signature);
    return 0;
}