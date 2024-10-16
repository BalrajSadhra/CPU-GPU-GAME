CC = gcc
OBJ_DIR = obj
OUT_DIR = build
LIB = -L"lib" -L"lib/x86_64"
INC = -I"include"

ifeq ($(OS),Windows_NT) 
	MK_OBJ_DIR = if not exist $(OBJ_DIR) md $(OBJ_DIR)
	MK_OUT_DIR = if not exist $(OUT_DIR) md $(OUT_DIR)
	EXT = .exe
	RM = del /s /q
else 
	MK_OBJ_DIR = mkdir -p $(@D)
	MK_OUT_DIR = mkdir -p $(@D)
	RM = rm -rf
	EXT =
endif 

CFLAGS = \
	-g \
	-Wall \
	$(INC) \
	$(LIB)

#CFLAGS += -DENABLE_OPENCL # Comment out to disable OpenCL collision detection
CFLAGS += -DENABLE_DBG
CFLAGS += -DSDL_MAIN_HANDLED
CFLAGS += -DENABLE_GOD_MODE #comment out if you do not want to be invincible
CFLAGS += -DENABLE_PROFILING

SRCS = $(wildcard *.c)
OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SRCS))

LINKERS = -lSDL2main \
		  -lSDL2 \
		  -lSDL2_image \
		  -lSDL2_ttf \
		  -lOpenCL

TARGET = $(OUT_DIR)/mccd$(EXT)

$(TARGET): $(OBJS) | $(OUT_DIR)
	@echo Linking $(TARGET)...
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LINKERS)

$(OUT_DIR):
	@$(MK_OUT_DIR)

$(OBJ_DIR)/%.o: %.c
	@echo Compiling $<
	@$(MK_OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

all: $(TARGET)

clean:
	@$(RM) $(OBJ_DIR) 
	@$(RM) *$(EXT)