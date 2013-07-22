TARGET = facesnap
CUDA_PATH=/opt/cuda/lib64/

SRC_DIR = src
CUDA_SRC_DIR = cuda
OBJ_DIR = obj

CC = g++
NVCC = nvcc
RM = rm -f

NVCCFLAGS = -arch sm_20
CPPFLAGS = -Wall -march=native -mtune=generic -O2 -pipe -fstack-protector --param=ssp-buffer-size=4 -g -fvar-tracking-assignments -D_FORTIFY_SOURCE=2 -std=c++11 -I./$(CUDA_SRC_DIR)
LDFLAGS = -Wall -Wl,-O1,--sort-common,--as-needed,-z,relro
LDLIBS = -lm -lcuda -lcudart -L$(CUDA_PATH)

CPPFLAGS += `pkg-config --cflags opencv` 
LDLIBS += `pkg-config --libs opencv`

CPP_FILES = $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES = $(wildcard $(CUDA_SRC_DIR)/*.cu)

H_FILES = $(wildcard $(SRC_DIR)/*.h)
CU_H_FILES = $(wildcard $(CUDA_SRC_DIR)/*.cu.h)

OBJ_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.cu.o)))

OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS += $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

$(TARGET) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS) 

$(OBJ_DIR)/%.cu.o : $(CUDA_SRC_DIR)/%.cu $(CU_H_FILES)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(H_FILES)
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) $(TARGET) $(OBJ_DIR)/*.o
