#include <stdio.h>
#include <stdbool.h>
#ifdef ENABLE_OPENCL
#include "CL/cl.h"
#endif
#include "SDL2/SDL.h"
#include "SDL2/SDL_image.h"
#include "collision.h"
#include "glad/glad.h"
#include "utils.h"
#include <time.h>
#include <math.h>
#include "vector.h"
#include "SDL2/SDL_ttf.h"
#include <time.h>

#ifdef ENABLE_OPENCL
const char *minkowski_kernel_str =
    "   typedef struct {\n"
    "       int x;\n"
    "       int y;\n"
    "   } VECTOR;\n"

    "__kernel void calculate_minkowski_diff(__global VECTOR* set1, const int set1_size, __global VECTOR* set2, const int set2_size, __global VECTOR* result) {\n"
    
    "    int gid_set1 = get_global_id(0);\n"
    "    int gid_set2 = get_global_id(1);\n"
    "\n"

    "    int result_idx = gid_set1 * set2_size + gid_set2; \n"
    "\n"

    // "    printf(\"gid_set1 %d, gid_set2 %d\", gid_set1, gid_set2);\n"
    "    if (gid_set1 < set1_size && gid_set2 < set2_size) {\n"
    "\n"
    "       result[result_idx].x = set1[gid_set1].x - set2[gid_set2].x;\n"  
    "       result[result_idx].y = set1[gid_set1].y - set2[gid_set2].y;\n"
    "    }\n"
    "}\n";

const char *collision_kernel_str =
    "   typedef struct {\n"
    "       int x;\n"
    "       int y;\n"
    "   } VECTOR;\n"

    "   __kernel void is_colliding(__global VECTOR* result, const int result_size, __global int* colliding) {\n"
    "       int gid = get_global_id(0);\n"
    "       int vertices_count = result_size;\n"
    "       int counter = 0;\n"
    "       for (int i = 0; i < vertices_count; i++) {\n"
    "           VECTOR p1 = result[i];\n"
    "           VECTOR p2 = result[(i + 1) % vertices_count];\n"
    "           if ((0 < p1.y) != (0 < p2.y) && 0 < p1.x + ((-p1.y) / (p2.y - p1.y)) * (p2.x - p1.x))\n"
    "               counter++;\n"
    "       }\n"
    // "       printf(\"colliding[0] %d\", (counter % 2 == 1));        \n"
    "       colliding[0] = (counter % 2 == 1);\n"
    "   }\n";
#endif

#define RECTANGLE_WIDTH 20
#define RECTANGLE_HEIGHT 20
#define RECTANGLE_SPEED 4
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define DIFFICULTY_COUNT 100

POLYGON g_player;
POLYGON_LIST g_polygon_list;
#ifdef ENABLE_OPENCL
cl_context g_context;
#endif

enum Screen {
    MAIN_SCREEN,
    GAME_OVER_SCREEN,
    DIFFICULTY_SCREEN,
    GAME_SCREEN
};

static void init_player();
static void update_position(POLYGON* polygon); // take in pointer to polygon
static void draw_polygon(SDL_Renderer* renderer, VECTOR vertices[], int vertex_count, SDL_Color color);
static void sort_arr(POLYGON*** p_arr, int polygon_count);
static void sweep_and_prune(POLYGON** p_arr, int num_polygons, int *p_collision_ids[], int *p_num_collisions);
#ifdef ENABLE_OPENCL
static void update_polygon_buffers(cl_command_queue queue, int p_collision_ids[], int num_collisions);
#endif

int main(int argc, char *argv[])
{
    int status;
    int free_polygons = 0;
    bool running = true;
    uint32_t lastSpawnTime = SDL_GetTicks();
    int p_idx=0;
    int diff_count = 0;
    POLYGON temp[DIFFICULTY_COUNT];
    enum Screen currentScreen = MAIN_SCREEN;
    int score = 0;
    char scoreText[20];
    time_t startTime = time(NULL);
    time_t currentTime;
    double elapsedTime = 0.0;
    int afkTime = 0;
    double scoreIncreaseInterval = 1.0;
    SDL_Color SDL_WHITE = {255, 255, 255, 255};
    int colliding = 0;

#ifdef ENABLE_OPENCL
    int local_size = 0;
    cl_uint num_devices, num_platforms;
    cl_device_id device;
    cl_platform_id platform;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_kernel kernel_2;

    DBG_PRINT("Getting OpenCL platform IDs...\n");
    status = clGetPlatformIDs(1, &platform, &num_platforms);
    if (status != CL_SUCCESS) {
        printf("Error getting platform ID: %d\n", status);
        goto Out;
    }

    DBG_PRINT("Getting OpenCL device IDs...\n");
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (status != CL_SUCCESS) {
        printf("Error getting device ID: %d\n", status);
        goto Out;
    }

    DBG_PRINT("Creating OpenCL context...\n");
    g_context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        
        printf("Error creating context: %d\n", status);
        goto Out;
    }

    DBG_PRINT("Creating OpenCL command queue...\n");
    queue = clCreateCommandQueue(g_context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS) {
        printf("Error creating command queue: %d\n", status);
        goto Out;
    }

    DBG_PRINT("Creating OpenCL program with kernel source...\n");
    cl_program program = clCreateProgramWithSource(g_context, 1, &minkowski_kernel_str, NULL, NULL);
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Error building program: %d\n", status);
        goto Out;
    }

    DBG_PRINT("Creating OpenCL program with kernel source...\n");
    cl_program program_2 = clCreateProgramWithSource(g_context, 1, &collision_kernel_str, NULL, NULL);
    status = clBuildProgram(program_2, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        printf("Error building program: %d\n", status);
        goto Out;
    }

    DBG_PRINT("Creating OpenCL kernels...\n");
    kernel = clCreateKernel(program, "calculate_minkowski_diff", NULL);
    kernel_2 = clCreateKernel(program_2, "is_colliding", NULL);

#else
    DBG_PRINT("This application is running without an OpenCL kernel.\n");
#endif

    status = SDL_Init(SDL_INIT_VIDEO);
    if (status < 0)
    {
        DBG_PRINT("Error initializing SDL: %s\n", SDL_GetError());
        goto Out;
    }

    if (TTF_Init() == -1) {
        printf("Error initializing SDL TTF %s\n", TTF_GetError());
        goto Out;
    }

    SDL_Window *window = SDL_CreateWindow(
        "Multiple Compute Collision Detection Program",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_OPENGL
    );

    if (!window) {
        DBG_PRINT("Error creating SDL window: %s\n", SDL_GetError());
        goto Out;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);

    init_player();
    create_polygon_list(&g_polygon_list);
    add_polygon_to_list(&g_polygon_list, &g_player);

    SDL_Texture *title = IMG_LoadTexture(renderer, "sprites/title.png");
    if (title == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    SDL_Texture *texture = IMG_LoadTexture(renderer, "sprites/start.png");
    if (texture == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    SDL_Texture *menu = IMG_LoadTexture(renderer, "sprites/scroll.png");
    if (menu == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    SDL_Texture *hearts[3];
    for (int i = 0; i < 3; i++) {
        hearts[i] = IMG_LoadTexture(renderer, "sprites/heart.png");
        if (hearts[i] == NULL) {
            printf("Error loading texture %d: %s\n", i, SDL_GetError());
        }
    }

    SDL_Texture *easy = IMG_LoadTexture(renderer, "sprites/easy.png");
    if (easy == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    SDL_Texture *medium = IMG_LoadTexture(renderer, "sprites/medium.png");
    if (medium == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    SDL_Texture *hard = IMG_LoadTexture(renderer, "sprites/hard.png");
    if (hard == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    SDL_Texture *again = IMG_LoadTexture(renderer, "sprites/again.png");
    if (again == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    SDL_Texture *mainMenu = IMG_LoadTexture(renderer, "sprites/mainMenu.png");
    if (mainMenu == NULL) {
        printf("Error loading texture: %s\n", SDL_GetError());
    }

    sprintf(scoreText, "Score: %d", score);
    TTF_Font *font = TTF_OpenFont("arial.ttf", 25);
    if (!font) {
        printf("Error loading font: %s\n", TTF_GetError());
        goto Out; 
    }

    TTF_Font *titleFont = TTF_OpenFont("arial.ttf", 75);
    if (!titleFont) {
        printf("Error loading font: %s\n", TTF_GetError());
        goto Out;
    }

    TTF_Font *subtitleFont = TTF_OpenFont("arial.ttf", 40);
    if (!subtitleFont) {
        printf("Failed to load font! SDL_ttf Error: %s\n", TTF_GetError());
        goto Out;
    }


#ifdef ENABLE_OPENCL
    cl_mem results_buffer = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, sizeof(VECTOR) * MAX_VERTICES*MAX_VERTICES, NULL, &status);
    if (status != CL_SUCCESS) {
        DBG_PRINT("Error collisionResultsBuffer clCreateBuffer: %d\n", status);
    }

    cl_mem colliding_buffer = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &status);
    if (status != CL_SUCCESS) {
        DBG_PRINT("Error collisionResultsBuffer clCreateBuffer: %d\n", status);
    }

    clSetKernelArg(kernel, 4, sizeof(cl_mem), &results_buffer);
    clSetKernelArg(kernel_2, 0, sizeof(cl_mem), &results_buffer);
    clSetKernelArg(kernel_2, 2, sizeof(cl_mem), &colliding_buffer);
#endif

    while (running)
    {
        SDL_Event event;

        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
                running = false;
                break;

            case SDL_MOUSEBUTTONDOWN:
                if (currentScreen == MAIN_SCREEN) {
                    score = 0;
                    for(int i = 0; i<3; i++)
                        hearts[i] = IMG_LoadTexture(renderer, "sprites/heart.png");
                    int mouseX, mouseY;
                    SDL_GetMouseState(&mouseX, &mouseY);
                    SDL_Rect imageRect = {200, 300, 400, 50};
                    if (SDL_PointInRect(&(SDL_Point){mouseX, mouseY}, &imageRect)) {
                        currentScreen = DIFFICULTY_SCREEN;
                    }
                } else if (currentScreen == GAME_SCREEN) {
                    int mouseX, mouseY;
                    SDL_GetMouseState(&mouseX, &mouseY);
                    SDL_Rect imageRect = {725, 25, 50, 50};
                    if (SDL_PointInRect(&(SDL_Point){mouseX, mouseY}, &imageRect)) {
                        currentScreen = MAIN_SCREEN;
                        free_polygons = 1;
                    }
                } else if(currentScreen == DIFFICULTY_SCREEN){
                    score = 0;
                    int mouseX, mouseY;
                    SDL_GetMouseState(&mouseX, &mouseY);

                    SDL_Rect easyRect = {200, 300, 400, 50};
                    SDL_Rect medRect = {200, 400, 400, 50};
                    SDL_Rect hardRect = {200, 500, 400, 50};

                    if (SDL_PointInRect(&(SDL_Point){mouseX, mouseY}, &easyRect)) {
                        currentScreen = GAME_SCREEN;
                        diff_count = 15;
                    } else if(SDL_PointInRect(&(SDL_Point){mouseX, mouseY}, &medRect)){
                        currentScreen = GAME_SCREEN;
                        diff_count = 20;
                    } else if(SDL_PointInRect(&(SDL_Point){mouseX, mouseY}, &hardRect)){
                        currentScreen = GAME_SCREEN;
                        diff_count = 25;
                    }
                } else if(currentScreen == GAME_OVER_SCREEN){
                    for(int i = 0; i<3; i++)
                        hearts[i] = IMG_LoadTexture(renderer, "sprites/heart.png");
                    free_polygons = 1;
                    int mouseX, mouseY;
                    SDL_GetMouseState(&mouseX, &mouseY);
                    
                    SDL_Rect mainMenuRect = {200, 300, 400, 50};
                    SDL_Rect diffMenuRect = {200, 400, 400, 50};

                    if (SDL_PointInRect(&(SDL_Point){mouseX, mouseY}, &mainMenuRect)) {
                        currentScreen = MAIN_SCREEN;
                    } else if(SDL_PointInRect(&(SDL_Point){mouseX, mouseY}, &diffMenuRect)){
                        //currentScreen = DIFFICULTY_SCREEN;
                        score = 0;
                        currentScreen = GAME_SCREEN;
                    }
                }
                break;

            case SDL_KEYDOWN:
                switch (event.key.keysym.sym){
                    case SDLK_LEFT:
                        set_velocity_x(&g_player, -RECTANGLE_SPEED);
                        if(afkTime > 0){
                            afkTime-=1;
                        } else{
                            afkTime = 0;
                        }
                        break;
                    case SDLK_RIGHT:
                        set_velocity_x(&g_player, RECTANGLE_SPEED);
                        if(afkTime > 0){
                            afkTime-=1;
                        } else{
                            afkTime = 0;
                        }
                        break;
                    case SDLK_UP:
                        set_velocity_y(&g_player, -RECTANGLE_SPEED);
                        break;
                    case SDLK_DOWN:
                        set_velocity_y(&g_player, RECTANGLE_SPEED);
                        break;
                    default:
                        break;
                }
                break;
            case SDL_KEYUP:
                switch( event.key.keysym.sym ){
                    case SDLK_LEFT:
                        if( g_player.velocity.x < 0 )
                            g_player.velocity.x = 0;
                        break;
                    case SDLK_RIGHT:
                        if( g_player.velocity.x > 0 )
                            g_player.velocity.x = 0;
                        break;
                    case SDLK_UP:
                        if( g_player.velocity.y < 0 )
                            g_player.velocity.y = 0;
                        break;
                    case SDLK_DOWN:
                        if( g_player.velocity.y > 0 )
                            g_player.velocity.y = 0;
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
            }
        }
        SDL_SetRenderDrawColor(renderer, 23, 79, 38, 255);
        SDL_RenderClear(renderer);

        // delete all existing polygons and reinitiate them for next games
        if (free_polygons) {
                delete_all_polygons(&g_polygon_list);
                init_player();
                create_polygon_list(&g_polygon_list);
                add_polygon_to_list(&g_polygon_list, &g_player);
                free_polygons = 0;
        }

        if (currentScreen == MAIN_SCREEN) {
            SDL_Rect titleRect = {100, 50, 600, 150}; 
            SDL_RenderCopy(renderer, title, NULL, &titleRect);
            
            SDL_Rect startRect = {200, 300, 400, 50}; 
            SDL_RenderCopy(renderer, texture, NULL, &startRect);


        } else if (currentScreen == GAME_OVER_SCREEN){

            SDL_Surface *gameOverSurface = TTF_RenderText_Solid(titleFont, "GAME OVER", SDL_WHITE);
            SDL_Texture *gameOverTexture = SDL_CreateTextureFromSurface(renderer, gameOverSurface);

            SDL_Rect gameOverRect = {185, 20, gameOverSurface->w, gameOverSurface->h};
            SDL_RenderCopy(renderer, gameOverTexture, NULL, &gameOverRect);

            SDL_FreeSurface(gameOverSurface);
            SDL_DestroyTexture(gameOverTexture);

            sprintf(scoreText, "HERE IS YOUR SCORE: %d", score);
            SDL_Surface *scoreSurface = TTF_RenderText_Solid(subtitleFont, scoreText, SDL_WHITE);
            SDL_Texture *scoreTexture = SDL_CreateTextureFromSurface(renderer, scoreSurface);

            SDL_Rect scoreOverRect = {150, 200, scoreSurface->w, scoreSurface->h};
            SDL_RenderCopy(renderer, gameOverTexture, NULL, &scoreOverRect);

            SDL_FreeSurface(scoreSurface);
            SDL_DestroyTexture(scoreTexture);

            SDL_Rect mainMenuRect = {200, 300, 400, 50}; 
            SDL_RenderCopy(renderer, mainMenu, NULL, &mainMenuRect);

            SDL_Rect diffMenuRect = {200, 400, 400, 50}; 
            SDL_RenderCopy(renderer, again, NULL, &diffMenuRect);

        } else if (currentScreen == GAME_SCREEN) {
            currentTime = time(NULL);
            elapsedTime = difftime(currentTime, startTime);

            if (elapsedTime >= scoreIncreaseInterval)
            {
                score = score + 1;
                sprintf(scoreText, "Score: %d", score);
                startTime = currentTime;
            }

            // Stop Player from being AFK
            afkTime += 1;
            if(afkTime > 500 && p_idx < diff_count){ // maybe remove the p_idx < diff_count condition
                int direction = (rand() % 2 == 0) ? 1 : -1;
                create_circle(&temp[p_idx], g_player.vertices[0].x+10, WINDOW_HEIGHT, 20, MAX_VERTICES); 
                set_velocity_y(&temp[p_idx], (rand() % (RECTANGLE_SPEED*2) +RECTANGLE_SPEED) * direction);
                //set_velocity_y(&temp[p_idx], 4);
                add_polygon_to_list(&g_polygon_list, &temp[p_idx]);
                p_idx++;
                afkTime = 0;
            }

            // Couldn't get the score to display on the screen without creating a new surface and texture but, I am freeing them right after so should be fine
            SDL_Surface *textSurface = TTF_RenderText_Solid(font, scoreText, SDL_WHITE);
            SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);

            // render the updated text
            SDL_Rect textRect = {130, 23, textSurface->w, textSurface->h};
            SDL_RenderCopy(renderer, textTexture, NULL, &textRect);

            SDL_FreeSurface(textSurface);
            SDL_DestroyTexture(textTexture);

            if (SDL_GetTicks() - lastSpawnTime > 5000) {
                if(p_idx < diff_count){ 
                    int direction = (rand() % 2 == 0) ? 1 : -1;
                    // todo: change circles to spawn in random spots!
                    create_circle(&temp[p_idx], rand() % (WINDOW_WIDTH), WINDOW_HEIGHT/2, 20, MAX_VERTICES);
                    //set_velocity_x(&temp[p_idx], rand() % (RECTANGLE_SPEED*2) - 2);
                    set_velocity_y(&temp[p_idx], (rand() % (RECTANGLE_SPEED*2) +RECTANGLE_SPEED) * direction);
                    add_polygon_to_list(&g_polygon_list, &temp[p_idx]);
                    p_idx++;
                    lastSpawnTime = SDL_GetTicks();
                }
            }
#ifdef ENABLE_PROFILING
            //START TRACKING HERE
            long long start_time, end_time;
            long long opencl_start_time, opencl_end_time;
            long long copy_start_time, copy_end_time;
            long long sweep_start_time, sweep_end_time;
            double kernel_exe_time = 0.0, total_time = 0.0, copy_time = 0.0, sweep_time = 0.0;
#endif

            SDL_Color color;
            POLYGON* current = g_polygon_list.head;
            POLYGON** polygon_arr = NULL; // pointer to pointer of array
            int num_polygons = 0;
            int *p_potential_collision_ids = NULL;
            int num_potential_collisions = 0;
#ifdef ENABLE_PROFILING
            sweep_start_time = current_microseconds();
#endif
            if(current != NULL){
                convert_list_to_arr(current, &polygon_arr, &num_polygons); // pass in addr of pointer to pointer of array
                sort_arr(&polygon_arr, num_polygons); // pass in addr of array
            }
            sweep_and_prune(polygon_arr, num_polygons, &p_potential_collision_ids, &num_potential_collisions);
#ifdef ENABLE_PROFILING
            sweep_end_time = current_microseconds();
            sweep_time = (sweep_end_time - sweep_start_time);
            printf("time to sweep and prune %d polygons: %.4f us\n", num_polygons, sweep_time);
#endif

#ifdef ENABLE_OPENCL
#ifdef ENABLE_PROFILING
            copy_start_time = current_microseconds();
#endif
            update_polygon_buffers(queue, p_potential_collision_ids, num_potential_collisions);
#ifdef ENABLE_PROFILING
            copy_end_time = current_microseconds();
            copy_time = (copy_end_time - copy_start_time);
            printf("time to update %d polygon buffers: %.4f us\n", num_polygons, copy_time);
#endif
#endif
#ifdef ENABLE_PROFILING
            start_time = current_microseconds();
#endif
            while(current != NULL) {
                POLYGON* current_next = current->next;
#ifdef ENABLE_OPENCL
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &current->object_buffer);
                clSetKernelArg(kernel, 1, sizeof(int), &current->vertices_idx);
#endif
                while(current_next != NULL && is_polygon_id_in_arr(p_potential_collision_ids, num_potential_collisions, current->id)) {
                    colliding = 0;
                    //printf("current id %d is in the array\n", current->id);
#ifdef ENABLE_OPENCL
                    size_t global_size[2] = {ALIGN_32(current->vertices_idx), ALIGN_32(current_next->vertices_idx)};
                    size_t global_size_2[] = {ALIGN_32(current->vertices_idx) * ALIGN_32(current_next->vertices_idx)};
                    int result_size[] = {current->vertices_idx * current_next->vertices_idx};
                    clSetKernelArg(kernel, 2, sizeof(cl_mem), &current_next->object_buffer);
                    clSetKernelArg(kernel, 3, sizeof(int), &current_next->vertices_idx);

                    if(is_polygon_id_in_arr(p_potential_collision_ids, num_potential_collisions, current_next->id)){ 
#ifdef ENABLE_PROFILING
                        opencl_start_time = current_microseconds(); 
#endif
                        //printf("current_next id %d is in the array\n", current_next->id);
                        status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
                        if (status != CL_SUCCESS) {
                            DBG_PRINT("Error enqueueing kernel: %d\n", status);
                        }

                        clSetKernelArg(kernel_2, 1, sizeof(int), &result_size);
                        status = clEnqueueNDRangeKernel(queue, kernel_2, 1, NULL, &global_size_2, NULL, 0, NULL, NULL);
                        if (status != CL_SUCCESS) {
                            DBG_PRINT("Error enqueueing kernel_2: %d\n", status);
                        }

                        status = clEnqueueReadBuffer(queue, colliding_buffer, CL_TRUE, 0, sizeof(int), &colliding, 0, NULL, NULL);
                        if(status != CL_SUCCESS) {
                            printf("Error reading colliding buffer: %d\n", status);
                        }
#ifdef ENABLE_PROFILING
                        opencl_end_time = current_microseconds(); 
                        kernel_exe_time += (opencl_end_time - opencl_start_time);
#endif
                    }

                    if(colliding) {
#else
                    
                    VECTOR result[current->vertices_idx*current_next->vertices_idx];
                    memset(result, 0, sizeof(VECTOR) * current->vertices_idx*current_next->vertices_idx);
                    colliding = is_polygon_id_in_arr(p_potential_collision_ids, num_potential_collisions, current_next->id);
                    bool origin_in_polygon = false;
                    if(colliding){ //potential collision, actually
#ifdef ENABLE_PROFILING
                        opencl_start_time = current_microseconds();
#endif
                        calculate_minkowski_diff(current->vertices, current->vertices_idx, current_next->vertices, current_next->vertices_idx, result);
                        origin_in_polygon = is_colliding(result, current->vertices_idx*current_next->vertices_idx);
#ifdef ENABLE_PROFILING
                        opencl_end_time = current_microseconds();
                        kernel_exe_time += (opencl_end_time - opencl_start_time);
#endif
                    }
                    if(colliding && origin_in_polygon) {
#endif
                        // first vector of the resulting colliding polygons
                        VECTOR overlap_vec = {current->vertices[0].x-current_next->vertices[0].x, 
                                                current->vertices[0].y-current_next->vertices[0].y};


                        double separation_factor = 0.15;
                        for(int i = 0; i < current->vertices_idx; i++) {
                            current->vertices[i].x += overlap_vec.x * separation_factor;
                            current->vertices[i].y += overlap_vec.y * separation_factor;
                        }

                        for(int i = 0; i < current_next->vertices_idx; i++) {
                            current_next->vertices[i].x -= overlap_vec.x * separation_factor;
                            current_next->vertices[i].y -= overlap_vec.y * separation_factor;
                        }

                        if(current_next->id == 0 ) {
#ifndef ENABLE_GOD_MODE
                            delete_polygon(&g_polygon_list, current);
                            // lose a life
                            if(hearts[2] != NULL) {
                                hearts[2] = NULL;
                            } else if(hearts[1] != NULL) {
                                hearts[1] = NULL;
                            } else if(hearts[0] != NULL) {
                                hearts[0] = NULL;
                            }
#endif
                        }
                        break;
                    }
                        // printf("collision with id's %d %d\n", current->id, current_next->id);
                    current_next = current_next->next;
                }

                update_position(current);
                draw_polygon(renderer, current->vertices, current->vertices_idx, SDL_WHITE);
                current = current->next;
            }
#ifdef ENABLE_PROFILING
                end_time = current_microseconds();
                total_time = (end_time-start_time);
                printf("elapsed time for num polygons %d: %.4f us\n", num_polygons, total_time);
                printf("time to execute all kernels for num polygons %d: %.4f us\n", num_polygons, kernel_exe_time);
#endif
                //print_polygon_list_details(&g_polygon_list);
                SDL_Rect menuRect = {725, 25, 50, 50}; 
                SDL_RenderCopy(renderer, menu, NULL, &menuRect);

                for(int i = 0; i < 3; i++) {
                    if(hearts[i] != NULL) {
                        SDL_Rect heartRect = {25 + (i * 30), 25, 25, 25};
                        SDL_RenderCopy(renderer, hearts[i], NULL, &heartRect);
                    }
                }

                if(hearts[0] == NULL && hearts[1] == NULL && hearts[2] == NULL) {
                    currentScreen = GAME_OVER_SCREEN;
                }

        } else if(currentScreen == DIFFICULTY_SCREEN){
            SDL_Rect easyRect = {200, 300, 400, 50}; 
            SDL_RenderCopy(renderer, easy, NULL, &easyRect);

            SDL_Rect mediumRect = {200, 400, 400, 50}; 
            SDL_RenderCopy(renderer, medium, NULL, &mediumRect);

            SDL_Rect hardRect = {200, 500, 400, 50}; 
            SDL_RenderCopy(renderer, hard, NULL, &hardRect);
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    return 0;

Out:
    DBG_PRINT("Releasing resources...\n");
#ifdef ENABLE_OPENCL
    clFinish(queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(g_context);
#endif
    exit(-1);
}

static void init_player() {
    create_polygon(&g_player, 4);

    add_vertice(&g_player, 100, 100); // top left
    add_vertice(&g_player, 100 , 100 + RECTANGLE_HEIGHT); // bottom left
    add_vertice(&g_player, 100 + RECTANGLE_WIDTH, 100 + RECTANGLE_HEIGHT); // bottom right
    add_vertice(&g_player, 100 + RECTANGLE_WIDTH, 100); // bottom right

}

static void draw_polygon(SDL_Renderer* renderer, VECTOR vertices[], int vertices_count, SDL_Color color) {
    int i;
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);

    for (i = 0; i < vertices_count - 1; i++) {
        SDL_RenderDrawLine(renderer, 
                            (int)vertices[i].x,
                            (int)vertices[i].y,
                            (int)vertices[i + 1].x, 
                            (int)vertices[i + 1].y);                    
    }
    SDL_RenderDrawLine(renderer, 
                        (int)vertices[i].x, 
                        (int)vertices[i].y,
                        (int)vertices[0].x, 
                        (int)vertices[0].y);
}

static void update_position(POLYGON* polygon) {

    int x_wrap = 0, y_wrap = 0;
    double min_x = polygon->vertices[0].x;
    double min_y = polygon->vertices[0].y;
    double max_x = polygon->vertices[0].x;
    double max_y = polygon->vertices[0].y;

    for (int i = 0; i < polygon->vertices_idx; i++) {
        min_x = fmin(min_x, polygon->vertices[i].x);
        min_y = fmin(min_y, polygon->vertices[i].y);
        max_x = fmax(max_x, polygon->vertices[i].x);
        max_y = fmax(max_y, polygon->vertices[i].y);
    }

    if (min_x < 0.0)
        x_wrap = WINDOW_WIDTH;
    if (max_x >= WINDOW_WIDTH)
        x_wrap = -WINDOW_WIDTH;
    if (min_y < 0.0)
        y_wrap = WINDOW_HEIGHT;
    if (max_y >= WINDOW_HEIGHT)
        y_wrap = -WINDOW_HEIGHT;

    for (int i = 0; i < polygon->vertices_idx; i++) {
        polygon->vertices[i].x += polygon->velocity.x + x_wrap;
        polygon->vertices[i].y += polygon->velocity.y + y_wrap;
    }

}

static void sort_arr(POLYGON*** p_arr, int polygon_count){ // takes in pointer to arr
    for (int i = 0; i < polygon_count - 1; i++) {
        for (int j = 0; j < polygon_count - i - 1; j++) {
            double min_x_1 = (*p_arr)[j]->vertices[0].x;
            double min_x_2 = (*p_arr)[j+1]->vertices[0].x;
            for (int k = 0; k < (*p_arr)[j]->vertices_idx; k++) {
                min_x_1 = fmin(min_x_1, (*p_arr)[j]->vertices[k].x);
            }
            for (int k = 0; k < (*p_arr)[j+1]->vertices_idx; k++) {
                min_x_2 = fmin(min_x_2, (*p_arr)[j+1]->vertices[k].x);
            }

            if (min_x_1 > min_x_2) {
                POLYGON* temp = (*p_arr)[j];
                (*p_arr)[j] = (*p_arr)[j + 1];
                (*p_arr)[j + 1] = temp;
            }
        }
    }
}

static void sweep_and_prune(POLYGON** p_arr, int num_polygons, int *p_collision_ids[], int *p_num_collisions){
    int *collision_arr = NULL;
    int collision_idx = 0;
    for(int i=0; i< num_polygons - 1; i++) {
        double max_x_i = p_arr[i]->vertices[0].x;
        for (int k = 0; k < p_arr[i]->vertices_idx; k++) {
            max_x_i = fmax(max_x_i, p_arr[i]->vertices[k].x);
        }

        for(int j=i+1; j< num_polygons; j++){
            double min_x_j = p_arr[j]->vertices[0].x;
            for (int k =0; k <p_arr[j]->vertices_idx; k++) {
                min_x_j = fmin(min_x_j, p_arr[j]->vertices[k].x);
            }
            if(min_x_j> max_x_i)
                break;

            if (!is_polygon_id_in_arr(collision_arr, collision_idx, p_arr[i]->id)) {
                collision_arr = (int*)realloc(collision_arr, (collision_idx + 1) * sizeof(int));
                collision_arr[collision_idx++] = p_arr[i]->id;
            }

            if (!is_polygon_id_in_arr(collision_arr, collision_idx, p_arr[j]->id)) {
                collision_arr = (int*)realloc(collision_arr, (collision_idx + 1) * sizeof(int));
                collision_arr[collision_idx++] = p_arr[j]->id;
            }

        }
    }
    
    *p_num_collisions = collision_idx;
    *p_collision_ids = collision_arr;
}

#ifdef ENABLE_OPENCL
static void update_polygon_buffers(cl_command_queue queue, int p_collision_ids[], int num_collisions) {
    int status = 0;
    POLYGON* current = g_polygon_list.head;
    cl_event map_event, unmap_event;
    while(current != NULL) {
        if(is_polygon_id_in_arr(p_collision_ids, num_collisions, current->id)){
            VECTOR* mapped_buffer = (VECTOR*)clEnqueueMapBuffer(queue, current->object_buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(VECTOR) * current->vertices_idx, 0, NULL, &map_event, &status);
            if (status != CL_SUCCESS) {
                DBG_PRINT("Error polygon ID %d clEnqueueMapBuffer: %d\n", current->id, status);
            }
            memcpy(mapped_buffer, current->vertices, sizeof(VECTOR)* current->vertices_idx);
            status = clEnqueueUnmapMemObject(queue, current->object_buffer, mapped_buffer, 0, NULL, &unmap_event);
            if (status != CL_SUCCESS) {
                DBG_PRINT("Error polygon ID %d clEnqueueUnmapMemObject: %d\n", current->id, status);
            }
        }
        current = current->next;
    }
    return;
}

#endif
