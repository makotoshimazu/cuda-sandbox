# Environment settings
CC ?= clang
CFLAGS ?= -std=c99 -O3 -Wextra -Wall

CXX ?= clang++
CXXFLAGS ?= -std=c++14 -O3 -Wextra -Wall

NVCC ?= /usr/local/cuda/bin/nvcc
HOSTCC ?= /usr/bin/g++-6
NVCFLAGS ?= -ccbin $(HOSTCC) -m64 -gencode arch=compute_70,code=sm_70 -O3

OUTDIR := out
SRCDIR := src

# Projects
PROJECTS := wmma vec_add

.PHONY: all
all: $(PROJECTS)

# ---------- wmma ----------
WMMA_SRCS_CPP := wmma/wmma.cpp
WMMA_SRCS_CU := wmma/kernel.cu
WMMA_OBJS = $(addprefix $(OUTDIR)/, $(WMMA_SRCS_CPP:.cpp=.cpp.o) $(WMMA_SRCS_CU:.cu=.cu.o))
WMMA_DEPS = $(WMMA_OBJS:.o=.d)

wmma: $(WMMA_OBJS)
	$(NVCC) $(NVCFLAGS) -o $@ $^

-include $(WMMA_DEPS)
# ---------------------------

# ---------- vec_add ----------
VEC_ADD_SRCS_CU := vec_add/vec_add.cu
VEC_ADD_OBJS = $(addprefix $(OUTDIR)/, $(VEC_ADD_SRCS_CU:.cu=.cu.o))
VEC_ADD_DEPS = $(VEC_ADD_OBJS:.o=.d)

vec_add: $(VEC_ADD_OBJS)
	$(NVCC) $(NVCFLAGS) -o $@ $^

-include $(VEC_ADD_DEPS)
# -----------------------------

# General rules
.SECONDARY:
$(OUTDIR)/%.cu.o: $(SRCDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

.SECONDARY:
$(OUTDIR)/%.cpp.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $<

.SECONDARY:
$(OUTDIR)/%.c.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -c -o $@ $<

.PHONY: clean
clean:
	rm -rf $(OUTDIR) $(PROJECTS)
