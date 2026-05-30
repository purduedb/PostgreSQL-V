# Build flags mirror pgvector_test.cpp.
PG_INCLUDE ?= $(shell pg_config --includedir)
CXX        ?= g++
CXXFLAGS   ?= -O3 -std=c++17 -mavx2 -mfma -fopenmp -Wall
INCLUDES    = -I$(PG_INCLUDE)
LIBS        = -lpq -lyaml-cpp -lcrypto -pthread

.PHONY: all test clean
all: replica_recall_eval replica_throughput_eval

test: test_runbook_common
	./test_runbook_common

test_runbook_common: test_runbook_common.cpp runbook_common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) test_runbook_common.cpp -o $@ $(LIBS)

replica_recall_eval: replica_recall_eval.cpp runbook_common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) replica_recall_eval.cpp -o $@ $(LIBS)

replica_throughput_eval: replica_throughput_eval.cpp runbook_common.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) replica_throughput_eval.cpp -o $@ $(LIBS)

clean:
	rm -f test_runbook_common replica_recall_eval replica_throughput_eval
