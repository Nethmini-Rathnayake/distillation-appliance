"""Hardware profiling and appliance-bandwidth adjustment.

Responsibilities:
    - Measure model size on disk and peak device memory.
    - Measure throughput on the dev machine, then scale to the target appliance
      profile: adjusted = measured * (target_bandwidth / dev_bandwidth).
      T4 (320 GB/s) -> appliance (273 GB/s) = 0.85.
    - Always emit both `measured_*` and `adjusted_*`, clearly labelled.
    - Assert the device actually in use. A silent CPU fallback invalidates every
      latency number, so it must be logged loudly.

Note: this module shadows the stdlib `profile` only inside the package
namespace; absolute imports elsewhere still resolve the stdlib module.
"""
