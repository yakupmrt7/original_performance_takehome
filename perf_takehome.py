"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots, vliw=False):
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def emit(self, bundle):
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space: {self.scratch_ptr} > {SCRATCH_SIZE}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        VL = VLEN  # 8
        n_groups = batch_size // VL  # 32

        # === SCRATCH ALLOCATION ===
        s_idx = self.alloc_scratch("s_idx", batch_size)   # 256 words for indices
        s_val = self.alloc_scratch("s_val", batch_size)   # 256 words for values

        # Vector temporaries - 3 sets minimum for pipeline interleaving
        s_nv = self.alloc_scratch("s_nv", VL)             # gathered node values
        s_vt1 = [self.alloc_scratch(f"svt1_{i}", VL) for i in range(3)]
        s_vt2 = [self.alloc_scratch(f"svt2_{i}", VL) for i in range(3)]
        s_idx2p1 = [self.alloc_scratch(f"si2p1_{i}", VL) for i in range(3)]
        s_vbit = self.alloc_scratch("s_vbit", VL)
        s_vidxn = self.alloc_scratch("s_vidxn", VL)
        s_vcmp = self.alloc_scratch("s_vcmp", VL)
        s_ga = self.alloc_scratch("s_ga", VL)             # 8 gather address regs

        # Vector constants
        s_v1 = self.alloc_scratch("s_v1", VL)
        s_v0 = self.alloc_scratch("s_v0", VL)
        s_vnn = self.alloc_scratch("s_vnn", VL)

        # Hash constant vectors
        s_hc = []
        s_hs = []
        for hi in range(len(HASH_STAGES)):
            s_hc.append(self.alloc_scratch(f"shc{hi}", VL))
            s_hs.append(self.alloc_scratch(f"shs{hi}", VL))

        # Scalar registers
        s_fvp = self.alloc_scratch("sfvp")
        s_iip = self.alloc_scratch("siip")
        s_ivp = self.alloc_scratch("sivp")
        s_one = self.alloc_scratch("sone")
        s_zero = self.alloc_scratch("szero")
        s_nn = self.alloc_scratch("snn")
        s_tmp = self.alloc_scratch("stmp")
        s_tmp2 = self.alloc_scratch("stmp2")
        s_addr = self.alloc_scratch("saddr")
        s_addr2 = self.alloc_scratch("saddr2")
        s_vlen_s = self.alloc_scratch("svlen")

        # === INITIALIZATION ===
        self.emit({"load": [("const", s_zero, 0), ("const", s_one, 1)]})
        self.emit({"load": [("const", s_tmp, 4), ("const", s_tmp2, 5)]})
        self.emit({"load": [("const", s_addr, 6), ("const", s_vlen_s, VL)]})
        self.emit({"load": [("load", s_nn, s_one), ("load", s_fvp, s_tmp)]})
        self.emit({"load": [("load", s_iip, s_tmp2), ("load", s_ivp, s_addr)]})

        # Broadcast vector constants + pipeline hash const loading
        # For stages 0,2,4: load multipliers (1 + 2^shift) instead of shifts
        stage0_mult = 1 + (1 << HASH_STAGES[0][4])  # 1 + 2^12 = 4097
        self.emit({"valu": [("vbroadcast", s_v1, s_one),
                            ("vbroadcast", s_v0, s_zero),
                            ("vbroadcast", s_vnn, s_nn)],
                   "load": [("const", s_tmp, HASH_STAGES[0][1] % (2**32)),
                            ("const", s_tmp2, stage0_mult)]})
        for hi in range(len(HASH_STAGES)):
            if hi + 1 < len(HASH_STAGES):
                next_val2 = HASH_STAGES[hi+1][4]
                # For stages 0,2,4: compute multiplier; for others: use shift value
                if hi+1 in [0, 2, 4] and HASH_STAGES[hi+1][0] == "+" and HASH_STAGES[hi+1][2] == "+" and HASH_STAGES[hi+1][3] == "<<":
                    next_val2 = 1 + (1 << HASH_STAGES[hi+1][4])
                self.emit({"valu": [("vbroadcast", s_hc[hi], s_tmp),
                                    ("vbroadcast", s_hs[hi], s_tmp2)],
                           "load": [("const", s_tmp, HASH_STAGES[hi+1][1] % (2**32)),
                                    ("const", s_tmp2, next_val2)]})
            else:
                self.emit({"valu": [("vbroadcast", s_hc[hi], s_tmp),
                                    ("vbroadcast", s_hs[hi], s_tmp2)]})

        # Load indices and values from memory into scratch
        self.emit({"alu": [("+", s_addr, s_iip, s_zero),
                           ("+", s_addr2, s_ivp, s_zero)]})
        for g in range(n_groups):
            bundle = {"load": [("vload", s_idx + g * VL, s_addr),
                               ("vload", s_val + g * VL, s_addr2)]}
            if g < n_groups - 1:
                bundle["alu"] = [("+", s_addr, s_addr, s_vlen_s),
                                 ("+", s_addr2, s_addr2, s_vlen_s)]
            self.emit(bundle)

        # Pause for reference kernel yield sync (only for local tests, disabled in submission)
        self.emit({"flow": [("pause",)]})

        # === UNROLLED ROUNDS with overlap (submission tests have pause disabled) ===
        # Each round pipeline is 143 cycles (groups 0-31 at spacing 4, last ends at 124+19=143)
        # Space rounds 128 cycles apart to avoid load conflicts (group 31 loads at 125-128)
        # This gives (143-128)*15 = 225 cycle savings vs sequential
        self._emit_overlapped_rounds(
            rounds, n_groups, VL, s_idx, s_val, s_nv, s_vt1, s_vt2,
            s_idx2p1, s_vbit, s_vidxn, s_vcmp, s_ga,
            s_v1, s_v0, s_vnn, s_hc, s_hs, s_fvp)

        # === STORE results back to memory (2 vstores per cycle) ===
        self.emit({"alu": [("+", s_addr, s_ivp, s_zero),
                           ("+", s_addr2, s_ivp, s_vlen_s),
                           ("+", s_tmp, s_vlen_s, s_vlen_s)]})  # s_addr2 = ivp+VL, s_tmp = 2*VL
        for g in range(0, n_groups, 2):
            bundle = {"store": [("vstore", s_addr, s_val + g * VL),
                                ("vstore", s_addr2, s_val + (g + 1) * VL)]}
            if g + 2 < n_groups:
                bundle["alu"] = [("+", s_addr, s_addr, s_tmp),
                                 ("+", s_addr2, s_addr2, s_tmp)]
            self.emit(bundle)

        self.emit({"flow": [("pause",)]})

    def _emit_pipelined_groups(self, n_groups, VL, s_idx, s_val, s_nv,
                                s_vt1, s_vt2, s_idx2p1, s_vbit, s_vidxn,
                                s_vcmp, s_ga, s_v1, s_v0, s_vnn, s_hc, s_hs, s_fvp):
        """
        Emit pipelined instructions for all groups.
        Overlaps gather of next group with hash of current group.
        Uses 5 sets of hash temporaries alternating by group.
        Pipeline spacing: 4 cycles between consecutive groups.
        Hash optimized with multiply_add for stages 0,2,4 reducing from 12 to 9 cycles.
        """
        SPACING = 4

        def group_ops(g, start_cycle):
            vi = s_idx + g * VL
            vv = s_val + g * VL
            buf = g % 3
            vt1 = s_vt1[buf]
            vt2 = s_vt2[buf]
            vidx2p1 = s_idx2p1[buf]
            ops = []

            # Cycle 0: ALU addr compute (8 ALU slots)
            ops.append((start_cycle, "alu", [("+", s_ga + i, s_fvp, vi + i) for i in range(VL)]))
            # Cycles 1-4: Load gather (2 loads per cycle)
            for c in range(4):
                ops.append((start_cycle + 1 + c, "load", [
                    ("load", s_nv + 2*c, s_ga + 2*c),
                    ("load", s_nv + 2*c + 1, s_ga + 2*c + 1)
                ]))
            # Cycle 5: VALU XOR val ^= node_val
            ops.append((start_cycle + 5, "valu", [("^", vv, vv, s_nv)]))
            # Cycles 6-14: Hash (6 stages, stages 0,2,4 use multiply_add = 1 cycle, others = 2 cycles)
            # Stage 0: (a+c)+(a<<12) = a*4097+c, Stage 2: (a+c)+(a<<5) = a*33+c, Stage 4: (a+c)+(a<<3) = a*9+c
            cycle_offset = 6
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                # Use multiply_add for stages 0, 2, 4 (collapse 2 cycles to 1)
                if hi in [0, 2, 4] and op1 == "+" and op2 == "+" and op3 == "<<":
                    multiplier = 1 + (1 << val3)  # 1 + 2^shift
                    # Need a broadcast constant for the multiplier
                    s_mult = s_hs[hi]  # Reuse the shift constant slot for multiplier
                    ops.append((start_cycle + cycle_offset, "valu", [("multiply_add", vv, vv, s_mult, s_hc[hi])]))
                    cycle_offset += 1
                else:
                    valu_tmp = [(op1, vt1, vv, s_hc[hi]), (op3, vt2, vv, s_hs[hi])]
                    valu_val = [(op2, vv, vt1, vt2)]
                    # Last hash stage (hi=5, cycles 13-14): also compute idx2, idx2p1
                    if hi == 5:
                        valu_tmp.append(("+", vidx2p1, vi, vi))       # idx2 = idx + idx
                        valu_val.append(("+", vidx2p1, vidx2p1, s_v1)) # idx2p1 = idx2 + 1
                    ops.append((start_cycle + cycle_offset, "valu", valu_tmp))
                    ops.append((start_cycle + cycle_offset + 1, "valu", valu_val))
                    cycle_offset += 2
            # After hash completes (cycle_offset is now at end of hash)
            # Bit extraction and index update
            ops.append((start_cycle + cycle_offset, "valu", [("&", s_vbit, vv, s_v1)]))
            ops.append((start_cycle + cycle_offset + 1, "valu", [("+", s_vidxn, vidx2p1, s_vbit)]))
            ops.append((start_cycle + cycle_offset + 2, "valu", [("<", s_vcmp, s_vidxn, s_vnn)]))
            ops.append((start_cycle + cycle_offset + 3, "flow", [("vselect", vi, s_vcmp, s_vidxn, s_v0)]))
            return ops

        starts = [g * SPACING for g in range(n_groups)]

        all_ops = []
        for g in range(n_groups):
            all_ops.extend(group_ops(g, starts[g]))

        max_cycle = max(cycle for cycle, _, _ in all_ops) + 1
        schedule = [defaultdict(list) for _ in range(max_cycle)]
        for cycle, engine, slots in all_ops:
            schedule[cycle][engine].extend(slots)

        for cycle_bundle in schedule:
            if cycle_bundle:
                self.emit(dict(cycle_bundle))

    def _emit_overlapped_rounds(self, rounds, n_groups, VL, s_idx, s_val, s_nv,
                                 s_vt1, s_vt2, s_idx2p1, s_vbit, s_vidxn,
                                 s_vcmp, s_ga, s_v1, s_v0, s_vnn, s_hc, s_hs, s_fvp):
        """
        Emit all rounds unrolled with inter-round overlap.
        Round spacing D=128 cycles avoids load conflicts while maximizing overlap.
        Group 31 loads at cycles 125-128, next round's group 0 loads at 129-132.
        Each round takes 143 cycles (32 groups * 4 spacing + 19 final cycles).
        Total time: 143 + 15*128 = 2063 cycles instead of 16*143 = 2288 cycles.
        """
        SPACING = 4
        ROUND_DELAY = 128  # cycles between round starts

        def group_ops(g, start_cycle):
            vi = s_idx + g * VL
            vv = s_val + g * VL
            buf = g % 3
            vt1 = s_vt1[buf]
            vt2 = s_vt2[buf]
            vidx2p1 = s_idx2p1[buf]
            ops = []

            # Cycle 0: ALU addr compute (8 ALU slots)
            ops.append((start_cycle, "alu", [("+", s_ga + i, s_fvp, vi + i) for i in range(VL)]))
            # Cycles 1-4: Load gather (2 loads per cycle)
            for c in range(4):
                ops.append((start_cycle + 1 + c, "load", [
                    ("load", s_nv + 2*c, s_ga + 2*c),
                    ("load", s_nv + 2*c + 1, s_ga + 2*c + 1)
                ]))
            # Cycle 5: VALU XOR val ^= node_val
            ops.append((start_cycle + 5, "valu", [("^", vv, vv, s_nv)]))
            # Cycles 6-14: Hash (6 stages, stages 0,2,4 use multiply_add = 1 cycle, others = 2 cycles)
            cycle_offset = 6
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                if hi in [0, 2, 4] and op1 == "+" and op2 == "+" and op3 == "<<":
                    ops.append((start_cycle + cycle_offset, "valu", [("multiply_add", vv, vv, s_hs[hi], s_hc[hi])]))
                    cycle_offset += 1
                else:
                    valu_tmp = [(op1, vt1, vv, s_hc[hi]), (op3, vt2, vv, s_hs[hi])]
                    valu_val = [(op2, vv, vt1, vt2)]
                    if hi == 5:
                        valu_tmp.append(("+", vidx2p1, vi, vi))
                        valu_val.append(("+", vidx2p1, vidx2p1, s_v1))
                    ops.append((start_cycle + cycle_offset, "valu", valu_tmp))
                    ops.append((start_cycle + cycle_offset + 1, "valu", valu_val))
                    cycle_offset += 2
            # Cycle 15-18: bit extraction and index update
            ops.append((start_cycle + 15, "valu", [("&", s_vbit, vv, s_v1)]))
            ops.append((start_cycle + 16, "valu", [("+", s_vidxn, vidx2p1, s_vbit)]))
            ops.append((start_cycle + 17, "valu", [("<", s_vcmp, s_vidxn, s_vnn)]))
            ops.append((start_cycle + 18, "flow", [("vselect", vi, s_vcmp, s_vidxn, s_v0)]))
            return ops

        # Generate all operations for all rounds with overlap
        all_ops = []
        for r in range(rounds):
            round_start = r * ROUND_DELAY
            for g in range(n_groups):
                group_start = round_start + g * SPACING
                all_ops.extend(group_ops(g, group_start))

        # Schedule all operations
        max_cycle = max(cycle for cycle, _, _ in all_ops) + 1
        schedule = [defaultdict(list) for _ in range(max_cycle)]
        for cycle, engine, slots in all_ops:
            schedule[cycle][engine].extend(slots)

        # Emit the schedule
        for cycle_bundle in schedule:
            if cycle_bundle:
                self.emit(dict(cycle_bundle))


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()
