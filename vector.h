#ifndef VECTOR_H
#define VECTOR_H

#include <stdbool.h>
#include "SDL2/SDL.h"
#ifdef ENABLE_OPENCL
#include "CL/cl.h"
#endif
#define PI 3.14159265358979323846
#define MAX_VERTICES 30 // change this if we use more vertices in any one polygon

// 16 bytes per vertice
typedef struct {
    int x;
    int y;
} VECTOR;

//up to 30 vertices per polygon, so max 480 bytes per polygon. two polygons sent to kernel per iteration, so 960 bytes input to kernel
typedef struct {
    int id;
    VECTOR* vertices;
    size_t vertices_idx;
    VECTOR velocity;
#ifdef ENABLE_OPENCL
    cl_mem object_buffer;
#endif
    struct POLYGON* next;
} POLYGON;

// for resulting minkowski difference vector set, 900 vertices maximum so 900*16 = 14400 bytes output from kernel
typedef struct {
    POLYGON* head;
} POLYGON_LIST;

void create_polygon(POLYGON* polygon, int initial_size);
void create_polygon_list(POLYGON_LIST* list);
void add_polygon_to_list(POLYGON_LIST* list, POLYGON* polygon);
void add_vertice(POLYGON* polygon, double x, double y);
void set_velocity_x(POLYGON* polygon, double val);
void set_velocity_y(POLYGON* polygon, double val);
void delete_polygon(POLYGON_LIST* list, POLYGON* polygon);
void create_circle(POLYGON* polygon, double center_x, double center_y, double radius, int polygon_count);
double cross_multiply(VECTOR v1, VECTOR v2);
double dot_multiply(VECTOR v1, VECTOR v2);
void calculate_minkowski_diff(VECTOR set1[], int set1_size, VECTOR set2[], int set2_size, VECTOR result[]);
void delete_all_polygons(POLYGON_LIST* polygon_list);
void print_polygon_list_details(POLYGON_LIST* polygon_list);
int is_polygon_id_in_arr(int* arr, int size, int polygon_id);
void convert_list_to_arr(POLYGON* head, POLYGON*** p_arr, int* p_no_polygons);

#endif  // VECTOR_H
