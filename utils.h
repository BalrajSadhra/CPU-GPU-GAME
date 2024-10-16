#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <time.h>

#define ARRAY_SIZE(arr) (sizeof(arr)/sizeof(*(arr)))
#define ALIGN_32(value) (((value) + 31) & ~31)

#ifdef ENABLE_PROFILING
long long current_microseconds();
#endif

#ifdef ENABLE_DBG
#define DBG_PRINT(str, ...) \
    do { \
        printf(str, ##__VA_ARGS__); \
    } while(0)
#else
#define DBG_PRINT(str, ...) do {} while(0)
#endif


#endif  // UTILS_H