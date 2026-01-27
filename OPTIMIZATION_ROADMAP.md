
## 11. BREAKTHROUGH DISCOVERY: Level-Aware Optimization

### 11.1 The Key Insight

**All batch elements START at tree index 0 (root) and progress through tree levels together!**

After round R, elements are at approximately level R+1 (with some wrapping).

**Critical observation**: Elements at the SAME tree level access a CONTIGUOUS range of node values!

| Level L | Node Index Range | Range Size | vs Batch Size (256) |
|---------|-----------------|------------|---------------------|
| 0 | [0, 0] | 1 | Massive reuse! |
| 1 | [1, 2] | 2 | Massive reuse! |
| 2 | [3, 6] | 4 | Massive reuse! |
| 3 | [7, 14] | 8 | Massive reuse! |
| 4 | [15, 30] | 16 | High reuse |
| 5 | [31, 62] | 32 | High reuse |
| 6 | [63, 126] | 64 | Good reuse |
| 7 | [127, 254] | 128 | Some reuse |
| 8 | [255, 510] | 256 | At capacity |
| 9 | [511, 1022] | 512 | Scattered |
| 10 | [1023, 2046] | 1024 | Scattered |

### 11.2 The Optimization Strategy

**Instead of 256 gather operations per round, preload the entire level!**

**For early rounds (0-7):**
```
Round 0: Load tree[0] ONCE, broadcast to all 256 elements
         Savings: 128 load cycles → 1 vload + broadcast
         
Round 1: Load tree[1:2] (2 values), index into cached values
         Savings: 128 load cycles → 1 vload + indexing
         
Round 7: Load tree[127:254] (128 values = 16 vloads)
         Savings: 128 load cycles → 16 vloads + indexing
```

### 11.3 Implementation Approach

**Phase 1: Preload all levels 0-7 into scratch (once at start)**
```
Levels 0-7 total nodes: 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255 nodes
vloads needed: 32 (255/8 rounded up)
Cycles for preload: ~32 cycles (done ONCE)
```

**Phase 2: Rounds 0-7 use scratch lookup instead of gather**
```
Per round: Need to compute offset into cached level
offset = idx - (2^level - 1)  # Offset within level

Then use scratch-based vector selection/gather
This requires:
1. Compute offset for each element
2. Select from cached values based on offset
```

**Phase 3: Rounds 8-15 use traditional gather (elements scattered)**

### 11.4 Estimated Savings

| Round | Current (gathers) | Optimized | Savings |
|-------|------------------|-----------|---------|
| 0-7 | 128 × 8 = 1024 | ~200* | ~824 |
| 8-15 | 128 × 8 = 1024 | 1024 | 0 |
| **Total** | **2048** | **~1224** | **~824** |

*Estimate includes level preloading and scratch indexing overhead

**Projected new total: 2127 - 824 = ~1303 cycles**

This would beat the 1363 target!

### 11.5 Challenges

1. **Scratch indexing**: Need to compute and use variable offsets into scratch
2. **Level tracking**: Need to know which level each element is at
3. **Wrap handling**: Elements that wrap to level 0 need special handling
4. **Bounds checking**: Still need to handle out-of-bounds indices

### 11.6 Alternative: Hybrid Approach

Instead of fully level-aware processing:
1. **Cache top 7 levels** (63 nodes) in scratch at startup
2. **Check if idx < 63** before each gather
3. **Use cached value OR gather** based on check

This adds overhead but captures most early-round savings.

---

## Appendix A: Quick Reference

### Slot Limits
```
alu: 12, valu: 6, load: 2, store: 2, flow: 1
```

### Key Constants
```
VLEN = 8
batch_size = 256
n_groups = 32
rounds = 16
tree_height = 10
n_nodes = 2047
```

### Cycle Budget
```
Init:  ~46 cycles
Main:  ~2063 cycles (target: <1700)
Store: ~17 cycles
Total: ~2127 cycles (target: <1790)
```

### Dependency Chain (Critical Path)
```
gather(4) → XOR(1) → hash(9) → index_update(4) → total = 18 cycles per group
+ address_compute(1) = 19 cycles
```

The 19-cycle per-group latency with 4-cycle spacing and 128-cycle round delay is near-optimal for this algorithm on this hardware.
