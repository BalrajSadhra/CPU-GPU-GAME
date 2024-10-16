#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "vector.h"
#include "utils.h"
#include <time.h>
int g_id;
#ifdef ENABLE_OPENCL
extern cl_context g_context;
#endif

void create_polygon(POLYGON* polygon, int num_vertices){
    polygon->vertices = (VECTOR*)malloc(num_vertices * sizeof(VECTOR));
    
    polygon->vertices_idx = 0;
    polygon->velocity.x = 0.0;
    polygon->velocity.y = 0.0;
    polygon->next = NULL;
    polygon->id = g_id++;
#ifdef ENABLE_OPENCL
    int status = 0;
    polygon->object_buffer = clCreateBuffer(g_context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY, sizeof(VECTOR) * MAX_VERTICES, NULL, &status);
    if (status != CL_SUCCESS)
        printf("Error creating buffer: %d\n", status);
#endif
}

void create_polygon_list(POLYGON_LIST* list){
    list = (POLYGON_LIST*)malloc(sizeof(POLYGON_LIST));
    list->head = NULL;
}

void add_polygon_to_list(POLYGON_LIST* list, POLYGON* polygon){
    polygon->next = list->head;
    list->head = polygon;
}

void delete_all_polygons(POLYGON_LIST* list) {
    POLYGON* current = list->head;
    while (current != NULL) {
        POLYGON* next = current->next;
        delete_polygon(list, current);
        current = next;
    }
    list->head = NULL;
    g_id = 0;
}

void add_vertice(POLYGON* polygon, double x, double y){

    if(polygon->vertices_idx % 4 == 0)
        polygon->vertices = (VECTOR*)realloc(polygon->vertices, (polygon->vertices_idx + 4) * sizeof(VECTOR));

    polygon->vertices[polygon->vertices_idx].x = x;
    polygon->vertices[polygon->vertices_idx].y = y;
    polygon->vertices_idx++;

}

void set_velocity_x(POLYGON* polygon, double val){
    polygon->velocity.x = val;
}

void set_velocity_y(POLYGON* polygon, double val){
    polygon->velocity.y = val;
}

void delete_polygon(POLYGON_LIST* list, POLYGON* polygon){
    POLYGON* current = list->head;
    POLYGON* prev = NULL;
    while (current != NULL && current->id != polygon->id) {
        prev = current;
        current = current->next;
    }

    if(current->id == polygon->id){
        if(prev != NULL)
            prev->next = current->next;
        else
            list->head = current->next;
    }
    memset(&polygon, 0, sizeof(polygon));
    free(polygon);
}

void create_circle(POLYGON* polygon, double center_x, double center_y, double radius, int polygon_count) {
    const double increments = 2 * PI / polygon_count;

    create_polygon(polygon, polygon_count);
    set_velocity_x(polygon, 0);
    set_velocity_y(polygon, 0);

    for (int i = 0; i < polygon_count; i++) {
        double angle = i * increments;
        double x = center_x + radius * cos(angle);
        double y = center_y + radius * sin(angle);
        add_vertice(polygon, x, y);
    }
}

double cross_multiply(VECTOR v1, VECTOR v2){
    return v1.x * v2.y - v1.y * v2.x;
}

double dot_multiply(VECTOR v1, VECTOR v2){
    return v1.x * v2.x + v1.y * v2.y;
}

void calculate_minkowski_diff(VECTOR set1[], int set1_size, VECTOR set2[], int set2_size, VECTOR result[]) {
#ifdef ENABLE_PROFILING
    // //START TRACKING HERE
    long long start_time, end_time;
    double time = 0.0;
    start_time = current_microseconds();
#endif
    int vertice = 0;

    for (int i = 0; i < set1_size; i++) {
        for (int j = 0; j < set2_size; j++) {
            result[vertice].x = set1[i].x - set2[j].x;
            result[vertice].y = set1[i].y - set2[j].y;
            vertice++;
        }
    }
#ifdef ENABLE_PROFILING
    end_time = current_microseconds();
    time = (end_time-start_time);
    printf("elapsed time of minkowski_diff : %.4f us\n", time);
#endif
}

int is_polygon_id_in_arr(int* arr, int size, int polygon_id) {
    for(int i = 0; i < size; i++) {
        if(arr[i] == polygon_id)
            return true;
    }
    return false;
}

void convert_list_to_arr(POLYGON* head, POLYGON*** p_arr, int* p_no_polygons){ // take in a pointer to polygon array
    int polygon_count = 0;
    POLYGON* current = head;

    while(current != NULL){
        current = current->next;
        polygon_count++;
    }

    *p_no_polygons = polygon_count;
    *p_arr = (POLYGON**)malloc(polygon_count * sizeof(POLYGON*)); // dereference (take value aka pointer to pointer of array, and allocate space for array)

    current = head;
    for(int i=0; i < polygon_count; i++) {
        if(current != NULL)
            (*p_arr)[i] = current; // set pointer of arr[i] to current polygon
        current = current->next;
    }

}
