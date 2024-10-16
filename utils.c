#include <time.h>
#include <stdio.h>

long long current_microseconds() {
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return ((long long)tp.tv_sec * 1000000LL + tp.tv_nsec / 1000);
}
