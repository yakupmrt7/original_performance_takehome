# Anthropic's Performance Take-Home Problem Description

## Overview

This is Anthropic's original performance engineering take-home challenge. The goal is to optimize a kernel running on a custom **VLIW SIMD architecture simulator** to minimize the number of clock cycles required for execution.

## The Challenge

**Objective:** Optimize the `KernelBuilder.build_kernel()` method in `perf_takehome.py` to achieve the lowest possible cycle count.

**Validation:** Run `python tests/submission_tests.py` to verify correctness and see your cycle count.

**Baseline:** The unoptimized starter code runs in **147,734 cycles**.

## Performance Benchmarks

| Cycles | Achievement |
|--------|-------------|
| 147,734 | Baseline (starter code) |
| 18,532 | Updated take-home starting point |
| 2,164 | Claude Opus 4 after many hours |
| 1,790 | Claude Opus 4.5 casual session (matches best human 2hr performance) |
| 1,579 | Claude Opus 4.5 after 2 hours |
| 1,548 | Claude Sonnet 4.5 after many hours |
| 1,487 | Claude Opus 4.5 after 11.5 hours |
| 1,363 | Claude Opus 4.5 improved harness |
| ??? | Best human performance (undisclosed) |

## The Algorithm Being Optimized

The kernel performs a **parallel tree traversal with hashing**:

```
For each round in rounds:
    For each batch element i:
        1. Load current index and value for element i
        2. Load tree node value at current index
        3. XOR value with node value, then apply hash function
        4. Compute next index: 2*idx + (1 if hash is even else 2)
        5. Wrap index to 0 if it exceeds tree bounds
        6. Store updated index and value back to memory
```

### The Hash Function

A 6-stage 32-bit hash function with the following stages:

```python
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),  # a = (a + const) + (a << 12)
    ("^", 0xC761C23C, "^", ">>", 19),  # a = (a ^ const) ^ (a >> 19)
    ("+", 0x165667B1, "+", "<<", 5),   # a = (a + const) + (a << 5)
    ("+", 0xD3A2646C, "^", "<<", 9),   # a = (a + const) ^ (a << 9)
    ("+", 0xFD7046C5, "+", "<<", 3),   # a = (a + const) + (a << 3)
    ("^", 0xB55A4F09, "^", ">>", 16),  # a = (a ^ const) ^ (a >> 16)
]
```

Each stage: `a = op2(op1(a, val1), op3(a, val3))`

### Data Structures

**Tree:** An implicit perfect balanced binary tree stored as a flat array of node values.

**Input:**
- `indices[]`: Current tree positions for each batch element
- `values[]`: Current hash values for each batch element
- `rounds`: Number of iterations to perform

**Test Parameters:** `forest_height=10`, `rounds=16`, `batch_size=256`

## The Machine Architecture

### VLIW (Very Long Instruction Word)

Each instruction bundle can execute multiple operations in parallel across different "engines":

| Engine | Slots per Cycle | Purpose |
|--------|-----------------|---------|
| `alu` | 12 | Scalar arithmetic operations |
| `valu` | 6 | Vector arithmetic (SIMD) |
| `load` | 2 | Load from memory / load constants |
| `store` | 2 | Store to memory |
| `flow` | 1 | Control flow, select operations |
| `debug` | 64 | Debugging (ignored in submission) |

**Key Property:** All operations in a single instruction bundle execute simultaneously. Writes don't take effect until the end of the cycle.

### SIMD (Single Instruction Multiple Data)

- Vector length: `VLEN = 8` elements
- Vector operations operate on 8 contiguous 32-bit words at once

### Memory Model

- **Main Memory:** Flat array of 32-bit words containing tree data and input/output
- **Scratch Space:** 1536 words serving as registers/cache (manually managed)
- All arithmetic operates on scratch space; loads/stores transfer between memory and scratch

## Instruction Set Reference

### ALU Operations (`alu` engine)
```
(op, dest, a1, a2)  # dest = a1 op a2
```
Supported ops: `+`, `-`, `*`, `//`, `cdiv`, `^`, `&`, `|`, `<<`, `>>`, `%`, `<`, `==`

### Vector ALU Operations (`valu` engine)
```
("vbroadcast", dest, src)       # Broadcast scalar to vector
(op, dest, a1, a2)              # Element-wise vector operation
("multiply_add", dest, a, b, c) # dest[i] = a[i]*b[i] + c[i]
```

### Load Operations (`load` engine)
```
("load", dest, addr)            # Load mem[scratch[addr]] to scratch[dest]
("load_offset", dest, addr, offset)
("vload", dest, addr)           # Load 8 contiguous words (addr is scalar)
("const", dest, val)            # Load immediate value to scratch
```

### Store Operations (`store` engine)
```
("store", addr, src)            # Store scratch[src] to mem[scratch[addr]]
("vstore", addr, src)           # Store 8 contiguous words
```

### Flow Operations (`flow` engine)
```
("select", dest, cond, a, b)    # dest = a if cond != 0 else b
("vselect", dest, cond, a, b)   # Vector select
("add_imm", dest, a, imm)       # dest = a + immediate
("cond_jump", cond, addr)       # Jump if cond != 0
("cond_jump_rel", cond, offset) # Relative conditional jump
("jump", addr)                  # Unconditional jump
("jump_indirect", addr)         # Jump to scratch[addr]
("halt",)                       # Stop execution
("pause",)                      # Pause (for debugging)
("trace_write", val)            # Write to trace buffer
("coreid", dest)                # Get core ID
```

## Memory Layout

```
Address | Content
--------|--------
0       | rounds
1       | n_nodes (tree size)
2       | batch_size
3       | forest_height
4       | forest_values_p (pointer to tree data)
5       | inp_indices_p (pointer to indices array)
6       | inp_values_p (pointer to values array)
7       | extra_room pointer
7+      | Tree node values
...     | Input indices
...     | Input values
```

## Optimization Opportunities

The baseline implementation is intentionally naive:

1. **No VLIW utilization** - Only one operation per instruction bundle
2. **No SIMD vectorization** - All operations are scalar despite batch_size=256
3. **No instruction-level parallelism** - Operations that could run in parallel don't
4. **Inefficient memory access patterns** - No batching of loads/stores
5. **Loop unrolling potential** - Fully unrolled loops for all rounds and batch elements
6. **Redundant operations** - Constants loaded repeatedly

### Key Optimization Strategies

1. **Vectorization (SIMD):** Process 8 batch elements simultaneously using `valu`, `vload`, `vstore`
2. **VLIW Packing:** Fill instruction bundles with independent operations
3. **Software Pipelining:** Overlap loads with computation
4. **Loop Structures:** Use jumps instead of fully unrolling
5. **Constant Hoisting:** Load constants once, reuse from scratch
6. **Memory Access Optimization:** Batch loads/stores where possible

## Debugging Tools

### Trace Visualization

1. Run: `python perf_takehome.py Tests.test_kernel_trace`
2. Run: `python watch_trace.py` (opens browser)
3. Click "Open Perfetto" to visualize instruction execution

The trace shows:
- Each engine's slot utilization per cycle
- Which operations execute when
- Scratch space value changes

### Debug Instructions

- `("debug", ("compare", loc, key))` - Verify scalar value against reference
- `("debug", ("vcompare", loc, keys))` - Verify vector values against reference
- Debug instructions are ignored in submission tests

## Files Structure

| File | Purpose |
|------|---------|
| `perf_takehome.py` | Main file with `KernelBuilder` to optimize |
| `problem.py` | Machine simulator and reference kernel |
| `tests/submission_tests.py` | Official correctness and performance tests |
| `tests/frozen_problem.py` | Frozen copy of simulator for fair testing |
| `watch_trace.py` | Local server for trace visualization |
| `watch_trace.html` | Trace viewer UI |

## Important Rules

1. **Do NOT modify the `tests/` folder** - This will invalidate your submission
2. Verify with: `git diff origin/main tests/` (should be empty)
3. The official cycle count comes from `tests/submission_tests.py`
4. Multicore is disabled (`N_CORES = 1`) - don't try to enable it

## Validation Commands

```bash
# Verify tests folder is unchanged
git diff origin/main tests/

# Run submission tests and get cycle count
python tests/submission_tests.py
```
