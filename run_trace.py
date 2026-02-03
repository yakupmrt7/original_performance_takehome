#!/usr/bin/env python3
"""
Generate trace.json the same way submission_tests run the kernel:
one full run with trace=True. Run this, then run `python watch_trace.py`
and click "Open Perfetto" to view the timeline.
"""
import sys
sys.path.insert(0, "tests")

from frozen_problem import (
    Machine,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    N_CORES,
)
from perf_takehome import KernelBuilder

def main():
    forest_height, rounds, batch_size = 10, 16, 256
    print(f"Building: {forest_height=}, {rounds=}, {batch_size=}")
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(
        mem, kb.instrs, kb.debug_info(), n_cores=N_CORES,
        trace=True,
        value_trace={},
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    ok = machine.mem[inp_values_p : inp_values_p + len(inp.values)] == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    print("CYCLES:", machine.cycle)
    print("Correct:", ok)
    if not ok:
        sys.exit(1)
    print("Wrote trace.json — run `python watch_trace.py` and click Open Perfetto")

if __name__ == "__main__":
    main()
