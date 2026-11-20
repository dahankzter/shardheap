// Threads whose first allocator call happens in a pthread-key destructor
// (after the Rust thread-locals' destructors ran) must not strand a heap each:
// peak memory stays bounded however many such threads come and go.
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

static pthread_key_t key;
static void dtor(void *p) { free(p); }
static void *worker(void *p) { pthread_setspecific(key, p); return NULL; }

int main(void) {
    pthread_key_create(&key, dtor);
    for (int i = 0; i < 2000; i++) {
        pthread_t t;
        pthread_create(&t, NULL, worker, malloc(64));
        pthread_join(t, NULL);
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    if (ru.ru_maxrss > 256 * 1024) {
        fprintf(stderr, "tsd.c: peak %ld MiB for 2000 threads\n", ru.ru_maxrss / 1024);
        return 1;
    }
    puts("tsd.c: ok");
    return 0;
}
