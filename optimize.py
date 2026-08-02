#!/usr/bin/env python3
"""Source-checkout compatibility wrapper for ``motionkernel optimize``.

Usage:
    python optimize.py \\
      --fastvideo-checkout /path/to/FastVideo \\
      --model Lightricks/LTX-Video \\
      --workload workloads/ltx_480p.yaml \\
      --budget-hours 10 \\
      --output workspace/ltx

    # Validate the environment and exit without starting a campaign:
    python optimize.py ... --preflight-only

    # Offline smoke / tests (simulated stages):
    MOTIONKERNEL_SIMULATE=1 MOTIONKERNEL_SIMULATE_OUTCOME=promoted \\
      python optimize.py ... --budget-hours 1 --output workspace/v1-smoke
"""

from __future__ import annotations

from autokernel.optimize.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
