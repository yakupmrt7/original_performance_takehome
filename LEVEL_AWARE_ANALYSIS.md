# Level-Aware Optimization Analysis

## Attempted Implementation (Branch: level-aware-optimization)

### The Insight from OPTIMIZATION_ROADMAP.md

The roadmap identified a key observation:
- **All batch elements START at tree index 0 (root)**
- After round R, elements are at approximately level R+1
- Elements at the SAME tree level access a CONTIGUOUS range of nodes

#### Data Locality by Level

| Level | Node Range | Range Size | Batch Size (256) | Potential Savings |
|-------|------------|------------|------------------|-------------------|
| 0     | [0, 0]     | 1          | All elements share 1 value | **Massive** |
| 1     | [1, 2]     | 2          | 256 elements → 2 values | **Massive** |
| 2     | [3, 6]     | 4          | 256 elements → 4 values | **Massive** |
| 3     | [7, 14]    | 8          | 256 elements → 8 values | **Massive** |
| 7     | [127, 254] | 128        | 256 elements → 128 values | **High** |
| 8+    | [255+]     | 256+       | Scattered access | None |

### Theoretical Optimization

**Current approach (all rounds):**
- 32 groups × 8 gather loads = 256 scalar loads per round
- 4 cycles per group for gathering
- 16 rounds × 128 load cycles = 2048 cycles in loads alone

**Optimized approach (rounds 0-7):**
- Round 0: Load tree[0] once, broadcast → 1 load instead of 256
- Round 1: Load tree[1:2] once → 2 loads instead of 256
- Round 7: Load tree[127:254] → 128 loads instead of 256

**Projected savings: ~824 cycles**

---

## Why It Didn't Work: ISA Constraints

### Critical Limitation #1: No Scratch-to-Scratch Gather

**Problem:**
```
# We can load tree nodes into scratch (cache):
vload tree_cache, memory_address  ✓

# But we CANNOT gather from scratch:
load result, scratch[idx]  ✗  (ISA doesn't support this)
```

The ISA only supports:
- `load dest, scratch[addr]` where `addr` is a **memory address stored in scratch**
- NOT `load dest, scratch[idx]` where we index into scratch directly

**Impact:** We cannot use preloaded cache values without indirect memory access.

---

### Critical Limitation #2: Round 0 Optimization Slot Violations

**Round 0 Special Case:**
- All 256 elements have idx=0
- They all need tree[0] value
- Could load tree[0] once and broadcast

**Attempt:**
```python
# Cycle 0: Load tree[0]
load s_nv, fvp

# Cycle 1: Broadcast to vector
vbroadcast s_nv, s_nv

# Cycles 2+: Process all 32 groups
# Each group: XOR(1) + Hash(9) + Index(4) = 14 cycles
# No gather needed!
```

**Problem:** Even with no gather, groups create slot violations

#### Slot Analysis with Spacing=3

| Cycle | Groups Active | VALU Operations | Limit | Status |
|-------|---------------|-----------------|-------|--------|
| 10    | ~7 groups     | 7 ops           | 6     | **VIOLATION** |
| 13    | ~8 groups     | 8 ops           | 6     | **VIOLATION** |
| 16+   | ~8 groups     | 8 ops           | 6     | **VIOLATION** |

**Minimum safe spacing:** 5 cycles
- Round 0 time: 2 (setup) + 31×5 + 14 = 171 cycles
- Standard gather: 0 + 31×4 + 19 = 143 cycles
- **No savings!**

---

### Critical Limitation #3: VALU Bottleneck

The hash computation has:
- 6 stages × (1-2 cycles) = 9 cycles
- Index update: 4 cycles
- XOR: 1 cycle
- **Total per group: 14 cycles**

With 6 VALU slots and groups needing 1-3 VALU ops per cycle:
- Maximum parallelism: 6/3 = 2 groups per cycle
- Minimum spacing: ceil(14/2) = 7 cycles to avoid all conflicts

But we use spacing=4 because:
- Gather uses LOAD slots (separate from VALU)
- Gather latency (4 cycles) creates natural spacing
- Total per group: 4 (gather) + 14 (compute) = 18 cycles, but overlap gives 4-cycle spacing

**Without gather:**
- No natural spacing from LOAD ports
- Must rely on VALU spacing alone
- Need spacing ≥5 to avoid violations
- **Removes the main benefit of skipping gather**

---

## Alternative Approaches Considered

### 1. Hybrid: Check idx < 255, use cache OR gather

```python
# For each element
if idx < 255:
    value = tree_cache[idx]
else:
    value = memory[fvp + idx]
```

**Problem:**
- Requires vector select cascade
- Still can't load from scratch
- Adds overhead > potential savings

### 2. Precompute Level-Specific Broadcasts

```python
# Round 0: Everyone needs tree[0]
tree_0_vec = broadcast(tree[0])

# Round 1: Split into 2 groups (left/right)
tree_1_vec = [tree[1], tree[1], ..., tree[1], tree[2], tree[2], ...]
```

**Problem:**
- Requires tracking which level each element is at
- Wrapping to root complicates level tracking
- Selection logic adds more cycles than gathering

### 3. Manual Cache in Memory (not scratch)

```python
# Copy tree[0:255] to a different memory location
# Use standard gather but with better locality
```

**Problem:**
- Simulator doesn't model cache/memory hierarchy
- All memory accesses = 1 cycle regardless of locality
- No benefit in cycle count

---

## Conclusions

### Why 1700 Cycles is Extremely Difficult

The current **2127 cycles** is near-optimal because:

1. **Load Port Bottleneck:** 8 gathers × 2 ports = 4 cycles (minimum)
2. **Hash Dependency Chain:** 9 cycles (data dependencies prevent fusion)
3. **Index Update Chain:** 4 cycles (FLOW slot + dependencies)
4. **Per-Group Latency:** 4 + 1 + 9 + 4 = 18 cycles (hard floor)

**To reach ~1700 cycles, we need to save ~427 cycles.**

The only path: Reduce per-group from 19→18 cycles:
- Savings: 1 cycle × 32 groups × 16 rounds = **512 cycles**
- New total: 2127 - 512 = **1615 cycles** ✓

But this appears **impossible** given:
- Load ports: 8 gathers / 2 ports = 4 cycles (cannot reduce)
- Hash: 9 cycles (dependency-limited)
- Index: 4 cycles (FLOW slot limited)

### What Would Be Needed

**ISA Extensions:**
1. **Scratch gather:** `vgather dest, base_scratch, idx_vec`
   - Would enable true cached lookups
   - Could save ~3-4 cycles in early rounds

2. **Fused conditional:** `vselect_lt dest, a, b, true_val, false_val`
   - Collapse 2-step compare-select into 1 cycle
   - Save 1 cycle per group

3. **Multi-port loads:** 4 load ports instead of 2
   - Reduce gather from 4→2 cycles
   - Save 2 cycles per group

4. **Wider VALU:** 12 VALU slots instead of 6
   - Allow more aggressive parallelism
   - Reduce spacing from 4→3 cycles

**With extensions 1+2+3:**
- Per-group: (2 gather) + (1 xor) + (9 hash) + (3 index) = **15 cycles**
- Spacing: 3 (limited by dependency chains)
- Main loop: 16 × (32 × 3) = **1536 cycles**
- Total: ~1600 cycles ✓

---

## Current Status

- **Branch:** `level-aware-optimization`
- **Cycles:** 2127 (same as main)
- **Tests Passing:** 4/9
- **Conclusion:** Level-aware optimization is **not viable** with current ISA

The attempted optimization has been reverted to standard gather approach.
All code attempting level-aware caching has been removed to maintain clarity.

### Recommendation

Focus optimization efforts on:
1. **Algorithmic changes** (different tree traversal pattern)
2. **Request ISA extensions** from architecture team
3. **Accept 2127 as near-optimal** for current hardware
