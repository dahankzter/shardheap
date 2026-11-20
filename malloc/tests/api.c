// The C allocation API under LD_PRELOAD=libshardheap_malloc.so: edge cases
// and threads freeing each other's blocks. Exits non-zero on any failure.
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            fprintf(stderr, "api.c:%d: check failed: %s\n", __LINE__, #c); \
            exit(1);                                                     \
        }                                                                \
    } while (0)

static int aligned(void *p, size_t a) { return ((uintptr_t)p % a) == 0; }

#define THREADS 8
#define PER 100000
#define SHARED 4096
static void *volatile shared[SHARED];

static void *worker(void *arg) {
    unsigned s = (unsigned)(uintptr_t)arg * 2654435761u + 1;
    for (int i = 0; i < PER; i++) {
        s = s * 1103515245u + 12345u;
        size_t size = 1 + (s >> 8) % 2048;
        if ((s >> 4) % 64 == 0) size = 70000 + (s >> 8) % 200000;
        void *p = malloc(size);
        CHECK(p != NULL);
        memset(p, 0x5a, size < 64 ? size : 64);
        void *old = __atomic_exchange_n(&shared[(s >> 16) % SHARED], p, __ATOMIC_ACQ_REL);
        free(old); // usually another thread's block
    }
    return NULL;
}

int main(void) {
    // malloc(0), free(NULL), realloc(NULL, n)
    void *z1 = malloc(0), *z2 = malloc(0);
    CHECK(z1 && z2 && z1 != z2 && aligned(z1, 16) && aligned(z2, 16));
    free(z1);
    free(z2);
    free(NULL);
    void *r = realloc(NULL, 10);
    CHECK(r != NULL);
    free(r);

    // calloc: zeroed, overflow
    unsigned char *c = calloc(100, 10);
    CHECK(c != NULL);
    for (int i = 0; i < 1000; i++) CHECK(c[i] == 0);
    free(c);
    errno = 0;
    CHECK(calloc((size_t)1 << 40, (size_t)1 << 40) == NULL && errno == ENOMEM);

    // realloc(p, 0) frees and returns NULL; across small and large
    void *p0 = malloc(10);
    CHECK(realloc(p0, 0) == NULL);
    unsigned char *g = malloc(100);
    for (int i = 0; i < 100; i++) g[i] = (unsigned char)i;
    g = realloc(g, 1 << 20);
    CHECK(g != NULL);
    for (int i = 0; i < 100; i++) CHECK(g[i] == (unsigned char)i);
    g = realloc(g, 50);
    CHECK(g != NULL);
    for (int i = 0; i < 50; i++) CHECK(g[i] == (unsigned char)i);
    free(g);

    // aligned variants
    for (size_t a = 16; a <= ((size_t)8 << 20); a <<= 1) {
        void *p = NULL;
        CHECK(posix_memalign(&p, a, 100) == 0 && p && aligned(p, a));
        memset(p, 1, 100);
        CHECK(malloc_usable_size(p) >= 100);
        free(p);
    }
    void *bad = NULL;
    CHECK(posix_memalign(&bad, 24, 100) == EINVAL);
    void *aa = aligned_alloc(64, 100), *ma = memalign(4096, 10), *va = valloc(10),
         *pa = pvalloc(10);
    CHECK(aa && aligned(aa, 64) && ma && aligned(ma, 4096) && va && aligned(va, 4096) &&
          pa && aligned(pa, 4096));
    free(aa);
    free(ma);
    free(va);
    free(pa);

    // usable size, reallocarray overflow
    void *u = malloc(100);
    CHECK(malloc_usable_size(u) >= 100);
    free(u);
    errno = 0;
    CHECK(reallocarray(NULL, (size_t)1 << 40, (size_t)1 << 40) == NULL && errno == ENOMEM);
    CHECK(malloc_usable_size(NULL) == 0);
    malloc_trim(0);

    // threads freeing each other's blocks
    pthread_t t[THREADS];
    for (uintptr_t i = 0; i < THREADS; i++) CHECK(pthread_create(&t[i], NULL, worker, (void *)i) == 0);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    for (int i = 0; i < SHARED; i++) free(shared[i]);

    puts("api.c: ok");
    return 0;
}
