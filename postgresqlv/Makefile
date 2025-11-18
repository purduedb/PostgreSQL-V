EXTENSION = vector
EXTVERSION = 0.8.0

MODULE_big = vector
DATA = $(wildcard sql/*--*--*.sql)
DATA_built = sql/$(EXTENSION)--$(EXTVERSION).sql
OBJS = src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o \
	   src/lsmindex.o src/lsm_segment.o src/lsmbackground.o src/lsmindex_io.o src/ringbuffer.o src/tasksend.o src/vector_index_worker.o src/vectorindeximpl.o src/utils.o src/lsm_merge_worker.o src/folly_f14_link_alias.o
HEADERS = src/halfvec.h src/sparsevec.h src/vector.h

TESTS = $(wildcard test/sql/*.sql)
REGRESS = $(patsubst test/sql/%.sql,%,$(TESTS))
REGRESS_OPTS = --inputdir=test --load-extension=$(EXTENSION)

# To compile for portability, run: make OPTFLAGS=""
OPTFLAGS = -march=native

# Mac ARM doesn't always support -march=native
ifeq ($(shell uname -s), Darwin)
	ifeq ($(shell uname -p), arm)
		# no difference with -march=armv8.5-a
		OPTFLAGS =
	endif
endif

# PowerPC doesn't support -march=native
ifneq ($(filter ppc64%, $(shell uname -m)), )
	OPTFLAGS =
endif

# For auto-vectorization:
# - GCC (needs -ftree-vectorize OR -O3) - https://gcc.gnu.org/projects/tree-ssa/vectorization.html
# - Clang (could use pragma instead) - https://llvm.org/docs/Vectorizers.html
PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math

# Debug GCC auto-vectorization
# PG_CFLAGS += -fopt-info-vec

# Debug Clang auto-vectorization
# PG_CFLAGS += -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize

CXX = g++

# knowhere
KNOWHERE_LIBS := -L/home/liu4127/knowhere/build/Release -lknowhere -lknowhere_utils /home/liu4127/knowhere/build/Release/libfaiss.a

# Folly and nlohmann from Conan (your path)
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

CXXFLAGS += -std=c++17 $(OPTFLAGS) -O3 -Ofast -march=native -fopenmp \
			-ftree-vectorize -fassociative-math -fno-signed-zeros -fno-trapping-math

CPPFLAGS += $(GLOG_INC)

PG_CPPFLAGS += -I$(shell $(PG_CONFIG) --includedir-server)
PG_CPPFLAGS += -I$(shell $(PG_CONFIG) --includedir)/internal
PG_CPPFLAGS += $(NLOHMANN_INC) $(FOLLY_INC) $(GLOG_INC)
# PG_CPPFLAGS += -I/usr/local/include
PG_CPPFLAGS += -I/home/liu4127/knowhere/include -I/home/liu4127/knowhere/src -I/home/liu4127/knowhere/thirdparty
PG_CPPFLAGS += -DGLOG_NO_ABBREVIATED_SEVERITIES
SHLIB_LINK += $(KNOWHERE_LIBS) $(FOLLY_LIB) $(GLOG_LIB) \
              -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -lgflags -lpthread -lstdc++ -lc -lblas -fopenmp \
              -Wl,-rpath,/home/liu4127/knowhere/build/Release -Wl,-rpath,$(GLOG_LIBDIR)


%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC -c $< -o $@

all: sql/$(EXTENSION)--$(EXTVERSION).sql

sql/$(EXTENSION)--$(EXTVERSION).sql: sql/$(EXTENSION).sql
	cp $< $@

PG_CONFIG = /home/liu4127/postgresql/pg_build/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# for Mac
ifeq ($(PROVE),)
	PROVE = prove
endif

# for Postgres < 15
PROVE_FLAGS += -I ./test/perl

prove_installcheck:
	rm -rf $(CURDIR)/tmp_check
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='6$(DEF_PGPORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) $(if $(PROVE_TESTS),$(PROVE_TESTS),test/t/*.pl)

.PHONY: dist

dist:
	mkdir -p dist
	git archive --format zip --prefix=$(EXTENSION)-$(EXTVERSION)/ --output dist/$(EXTENSION)-$(EXTVERSION).zip master

# for Docker
PG_MAJOR ?= 17

.PHONY: docker

docker:
	docker build --pull --no-cache --build-arg PG_MAJOR=$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) .

.PHONY: docker-release

docker-release:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) --build-arg DEBIAN_CODENAME=bookworm -t pgvector/pgvector:pg$(PG_MAJOR) -t pgvector/pgvector:pg$(PG_MAJOR)-bookworm -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR) -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR)-bookworm .

.PHONY: docker-release-trixie

docker-release-trixie:
	docker buildx build --push --pull --no-cache --platform linux/amd64,linux/arm64 --build-arg PG_MAJOR=$(PG_MAJOR) --build-arg DEBIAN_CODENAME=trixie -t pgvector/pgvector:pg$(PG_MAJOR)-trixie -t pgvector/pgvector:$(EXTVERSION)-pg$(PG_MAJOR)-trixie .
