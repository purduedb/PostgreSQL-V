# Makefile for knowhere test programs

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -pthread

# knowhere paths (adjust these to match your setup)
KNOWHERE_BASE = /home/liu4127/knowhere
KNOWHERE_INC = -I$(KNOWHERE_BASE)/include -I$(KNOWHERE_BASE)/src -I$(KNOWHERE_BASE)/thirdparty
KNOWHERE_LIBDIR = $(KNOWHERE_BASE)/build/Release
KNOWHERE_LIBS = -L$(KNOWHERE_LIBDIR) -lknowhere -lknowhere_utils $(KNOWHERE_LIBDIR)/libfaiss.a

# Folly and nlohmann from Conan
NLOHMANN_INC := -I/home/liu4127/.conan/data/nlohmann_json/3.11.2/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include
FOLLY_BASE   := /home/liu4127/.conan/data/folly/2023.10.30.09/milvus/dev/package/763c5ce8d7ce4bf601cff416efb08f89b95b9205
FOLLY_INC    := -I$(FOLLY_BASE)/include
FOLLY_LIBDIR := $(FOLLY_BASE)/lib
FOLLY_LIB    := -L$(FOLLY_LIBDIR) -lfolly

# glog from Conan
GLOG_BASE   := /home/liu4127/.conan/data/glog/0.6.0/_/_/package/dd3c9e0dd79522c7df25a63d555c80dfef332d06
GLOG_INC    := -I$(GLOG_BASE)/include
GLOG_LIBDIR := $(GLOG_BASE)/lib
GLOG_LIB    := -L$(GLOG_LIBDIR) -lglog

CPPFLAGS = $(KNOWHERE_INC) $(NLOHMANN_INC) $(FOLLY_INC) $(GLOG_INC) -DGLOG_NO_ABBREVIATED_SEVERITIES
LDFLAGS = $(KNOWHERE_LIBS) $(FOLLY_LIB) $(GLOG_LIB) -lgflags -lpthread -lstdc++ -lc -lblas -fopenmp \
          -Wl,-rpath,$(KNOWHERE_LIBDIR) -Wl,-rpath,$(GLOG_LIBDIR)

# Targets
TARGET_CONCURRENT = knowhere_concurrent_query
SOURCE_CONCURRENT = knowhere_concurrent_query.cpp

TARGET_RANDOM = knowhere_concurrent_query_random
SOURCE_RANDOM = knowhere_concurrent_query_random.cpp

TARGET_HNSW_MMAP_PERF = hnsw_mmap_perf_test
SOURCE_HNSW_MMAP_PERF = hnsw_mmap_perf_test.cpp

all: $(TARGET_CONCURRENT) $(TARGET_RANDOM) $(TARGET_HNSW_MMAP_PERF)

$(TARGET_CONCURRENT): $(SOURCE_CONCURRENT)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(TARGET_CONCURRENT) $(SOURCE_CONCURRENT) $(LDFLAGS)

$(TARGET_RANDOM): $(SOURCE_RANDOM)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(TARGET_RANDOM) $(SOURCE_RANDOM) $(LDFLAGS)

$(TARGET_HNSW_MMAP_PERF): $(SOURCE_HNSW_MMAP_PERF)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(TARGET_HNSW_MMAP_PERF) $(SOURCE_HNSW_MMAP_PERF) $(LDFLAGS)

clean:
	rm -f $(TARGET_CONCURRENT) $(TARGET_RANDOM) $(TARGET_HNSW_MMAP_PERF)

.PHONY: all clean
