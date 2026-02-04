"""
Run the same simulation as submission_tests and produce a highly detailed
bottleneck report: where cycles are spent, which engines/slots have the most
overhead (idle capacity), and which parts consume the most cycles.

Usage:
  python profile_bottleneck.py

Uses frozen_problem (same as submission_tests) and perf_takehome.KernelBuilder.
"""

import os
import sys
import inspect
from collections import defaultdict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from functools import lru_cache

from tests.frozen_problem import (
    Machine,
    SLOT_LIMITS,
    CoreState,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    N_CORES,
)
from perf_takehome import KernelBuilder


# Engines that count toward cycles (exclude debug)
EXEC_ENGINES = ["alu", "valu", "load", "store", "flow"]


class InstrumentedMachine(Machine):
    """
    Machine subclass that records per-cycle engine/slot usage so we can
    report utilization and bottlenecks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-cycle: list of {engine: slots_used} (only exec engines)
        self._cycle_usage = []
        # Per-cycle, per-engine: list of slot indices used (e.g. valu -> [0,1,2,4] for slots 0,1,2,4)
        self._cycle_slot_detail = []

    def run(self):
        for core in self.cores:
            if core.state == CoreState.PAUSED:
                core.state = CoreState.RUNNING
        while any(c.state == CoreState.RUNNING for c in self.cores):
            has_non_debug = False
            usage = {}
            slot_detail = defaultdict(list)
            for core in self.cores:
                if core.state != CoreState.RUNNING:
                    continue
                if core.pc >= len(self.program):
                    core.state = CoreState.STOPPED
                    continue
                instr = self.program[core.pc]
                if self.prints:
                    self.print_step(instr, core)
                core.pc += 1
                # Record this instruction's engine usage before stepping
                for name, slots in instr.items():
                    if name == "debug":
                        continue
                    if name not in EXEC_ENGINES:
                        continue
                    n = len(slots)
                    usage[name] = usage.get(name, 0) + n
                    for i in range(n):
                        slot_detail[name].append(i)
                self.step(instr, core)
                if any(name != "debug" for name in instr.keys()):
                    has_non_debug = True
            if has_non_debug:
                self.cycle += 1
                self._cycle_usage.append(usage)
                self._cycle_slot_detail.append(dict(slot_detail))


@lru_cache(maxsize=None)
def kernel_builder(forest_height: int, n_nodes: int, batch_size: int, rounds: int):
    kb = KernelBuilder()
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)
    return kb


def run_profiled_simulation(forest_height: int, rounds: int, batch_size: int, seed: int = None):
    """Run the same test as submission_tests but with InstrumentedMachine. Returns (cycles, machine)."""
    if seed is not None:
        random = __import__("random")
        random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = kernel_builder(forest.height, len(forest.values), len(inp.indices), rounds)
    machine = InstrumentedMachine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    # Correctness check
    for ref_mem in reference_kernel2(mem):
        pass
    inp_values_p = ref_mem[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"

    return machine.cycle, machine


def print_bottleneck_report(cycles: int, machine: InstrumentedMachine):
    """Produce highly detailed bottleneck and overhead report."""
    total_cycles = cycles
    slot_limits = {k: v for k, v in SLOT_LIMITS.items() if k in EXEC_ENGINES}

    # Aggregate stats per engine
    total_slots_used = defaultdict(int)
    total_slot_capacity = {e: total_cycles * slot_limits[e] for e in EXEC_ENGINES}
    cycles_with_engine = defaultdict(int)
    # Histogram: for each engine, count of cycles with 0, 1, 2, ... slots used
    hist = {e: defaultdict(int) for e in EXEC_ENGINES}
    # Per-slot usage (for valu 0..5, load 0..1, etc.)
    slot_use_count = defaultdict(lambda: defaultdict(int))  # engine -> slot_index -> cycles used

    for cy, usage in enumerate(machine._cycle_usage):
        for e in EXEC_ENGINES:
            n = usage.get(e, 0)
            total_slots_used[e] += n
            if n > 0:
                cycles_with_engine[e] += 1
            hist[e][n] += 1
        detail = machine._cycle_slot_detail[cy] if cy < len(machine._cycle_slot_detail) else {}
        for e, slots in detail.items():
            for si in slots:
                slot_use_count[e][si] += 1

    # Wasted slot-cycles (bottleneck metric)
    wasted = {}
    utilization_pct = {}
    for e in EXEC_ENGINES:
        cap = total_slot_capacity[e]
        used = total_slots_used[e]
        wasted[e] = cap - used
        utilization_pct[e] = (100.0 * used / cap) if cap else 0

    # VALU slot 5 specific (the "VALU-5 gap")
    valu5_used = slot_use_count["valu"].get(5, 0)
    valu5_idle_cycles = total_cycles - valu5_used

    # --- Print report ---
    print()
    print("=" * 80)
    print("BOTTLENECK REPORT (same simulation as submission_tests)")
    print("=" * 80)
    print()
    print(f"Total cycles: {total_cycles}")
    print()

    print("-" * 80)
    print("1. PER-ENGINE UTILIZATION (higher = better, 100% = no idle slots)")
    print("-" * 80)
    for e in EXEC_ENGINES:
        limit = slot_limits[e]
        used = total_slots_used[e]
        cap = total_slot_capacity[e]
        waste = wasted[e]
        pct = utilization_pct[e]
        print(f"  {e:6}  slots/cycle limit={limit:2}  total_ops={used:6}  utilization={pct:5.1f}%  wasted_slot_cycles={waste:6}")
    print()

    print("-" * 80)
    print("2. WASTED CAPACITY (main overhead; higher = bigger bottleneck)")
    print("-" * 80)
    sorted_waste = sorted(wasted.items(), key=lambda x: -x[1])
    for e, w in sorted_waste:
        pct_of_total = 100.0 * w / (total_cycles * sum(slot_limits.values())) if total_cycles else 0
        print(f"  {e:6}  wasted_slot_cycles={w:6}  (~{pct_of_total:.1f}% of all slot capacity)")
    print()

    print("-" * 80)
    print("3. VALU SLOT BREAKDOWN (VALU-5 gap = slot 5 idle often)")
    print("-" * 80)
    for slot_i in range(slot_limits["valu"]):
        count = slot_use_count["valu"].get(slot_i, 0)
        pct = 100.0 * count / total_cycles if total_cycles else 0
        idle = total_cycles - count
        print(f"  VALU-{slot_i}  used in {count:5} cycles ({pct:5.1f}%)  idle in {idle:5} cycles")
    print(f"  --> VALU-5 idle cycles (gap): {valu5_idle_cycles} ({100.0 * valu5_idle_cycles / total_cycles:.1f}% of run)")
    print()

    print("-" * 80)
    print("4. LOAD / STORE SLOT BREAKDOWN")
    print("-" * 80)
    for e in ["load", "store"]:
        limit = slot_limits[e]
        for slot_i in range(limit):
            count = slot_use_count[e].get(slot_i, 0)
            pct = 100.0 * count / total_cycles if total_cycles else 0
            print(f"  {e}-{slot_i}  used in {count:5} cycles ({pct:5.1f}%)")
    print()

    print("-" * 80)
    print("5. HISTOGRAM: cycles by number of slots used per engine")
    print("   (e.g. many cycles with valu=5 means VALU-6th slot often idle)")
    print("-" * 80)
    for e in EXEC_ENGINES:
        limit = slot_limits[e]
        dist = [hist[e][n] for n in range(limit + 1)]
        bar = "  " + "  ".join(f"{n}:{hist[e][n]:5}" for n in range(limit + 1))
        print(f"  {e:6}  {bar}")
    print()

    print("-" * 80)
    print("6. CYCLES WITH IDLE SLOTS (cycles where engine had spare capacity)")
    print("-" * 80)
    for e in EXEC_ENGINES:
        limit = slot_limits[e]
        cycles_idle = sum(hist[e][n] for n in range(limit))
        cycles_full = hist[e][limit]
        pct_idle = 100.0 * cycles_idle / total_cycles if total_cycles else 0
        print(f"  {e:6}  full ({limit} slots): {cycles_full:5} cycles  |  at least one idle: {cycles_idle:5} cycles ({pct_idle:.1f}%)")
    print()

    # Cycles where VALU had exactly 5 or fewer (so at least one VALU slot idle)
    valu_not_full = sum(hist["valu"][n] for n in range(slot_limits["valu"]))
    print("-" * 80)
    print("7. SUMMARY: Top bottlenecks (by wasted slot-cycles)")
    print("-" * 80)
    for i, (e, w) in enumerate(sorted_waste[:5], 1):
        print(f"  #{i} {e}: {w} wasted slot-cycles (utilization {utilization_pct[e]:.1f}%)")
    print()
    print(f"  VALU-5 specific: {valu5_idle_cycles} cycles where slot 5 was idle.")
    print(f"  VALU not full (any slot idle): {valu_not_full} cycles ({100.0 * valu_not_full / total_cycles:.1f}%).")
    print()
    print("=" * 80)


def main():
    forest_height = 10
    rounds = 16
    batch_size = 256
    # Match submission_tests: no fixed seed (they say "random generator is not seeded")
    cycles, machine = run_profiled_simulation(forest_height, rounds, batch_size, seed=None)
    print(f"CYCLES: {cycles}")
    print_bottleneck_report(cycles, machine)


if __name__ == "__main__":
    main()
