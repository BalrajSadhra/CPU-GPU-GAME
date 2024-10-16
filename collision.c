#include <stdio.h>
#ifdef ENABLE_OPENCL
#include "CL/cl.h"
#endif
#include "collision.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

int is_colliding(VECTOR vertices[], int vertices_count) {
#ifdef ENABLE_PROFILING
    // //START TRACKING HERE
    long long start_time, end_time;
    double time = 0.0;
    start_time = current_microseconds();
#endif
    int i, counter = 0;
    VECTOR p1, p2;

    for (i = 0; i < vertices_count - 1; i++) {
        p1 = vertices[i];
        p2 = vertices[i+1]; 

        if( (0 < p1.y) != (0 < p2.y) && 0 < p1.x + ( (-p1.y)/(p2.y-p1.y) )*(p2.x-p1.x) )
            counter++;
    }

    p1 = vertices[i];
    p2 = vertices[0]; // this will connect the last point to the first point, completing the polygon.

        if( (0 < p1.y) != (0 < p2.y) && 0 < p1.x + ( (-p1.y)/(p2.y-p1.y) )*(p2.x-p1.x) )
            counter++;
#ifdef ENABLE_PROFILING
    end_time = current_microseconds();
    time = (end_time-start_time);
    printf("elapsed time of is_colliding : %.4f us\n", time);
#endif
    return (counter % 2 == 1);
}