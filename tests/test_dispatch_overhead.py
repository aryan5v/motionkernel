"""CPU tests for dispatch-overhead attribution and break-even analysis.

The fixtures are the real R4 timing reports: the eager-replay profile that
produced the 3.104 ms figure, and the mid-debugging CUDA-graph profile in
which every capture was declined ("runtime input 10 is bool, not a tensor").
"""

from __future__ import annotations

import json
import unittest

from autokernel.dispatch import (
    DispatchAnalysisError,
    TimingReport,
    attribute_overhead,
    breakeven_curve,
    overhead_from_e2e,
    required_saving_ms_per_call,
)

# profile-shadow/timing.json from the R4 run: eager FX replay, 1151 candidate
# calls, native forward shadowed on identical inputs.
R4_EAGER_PROFILE = {
    "timing_schema_version": 1,
    "synchronized": True,
    "phases": {
        "dispatch.candidate_total": {"calls": 1151, "mean_ms": 18.4899, "total_seconds": 21.281823},
        "dispatch.native_fallback": {"calls": 546, "mean_ms": 1.6377, "total_seconds": 0.894183},
        "dispatch.native_reference": {"calls": 64, "mean_ms": 2.6498, "total_seconds": 0.169589},
        "dispatch.shape_key": {"calls": 1761, "mean_ms": 0.1066, "total_seconds": 0.187727},
        "shadow.native_forward": {"calls": 1151, "mean_ms": 8.1758, "total_seconds": 9.410347},
        "subgraph.execute": {"calls": 1151, "mean_ms": 11.5666, "total_seconds": 13.313139},
        "subgraph.flatten": {"calls": 1151, "mean_ms": 0.0631, "total_seconds": 0.07265},
        "subgraph.unflatten": {"calls": 1151, "mean_ms": 0.044, "total_seconds": 0.050657},
        "subgraph.validate": {"calls": 1151, "mean_ms": 0.0447, "total_seconds": 0.051488},
    },
    "notes": {},
}

# profile-cudagraph/timing.json from the R4 run: every capture declined
# (bool runtime input), so 192 graph-path attempts (48 scopes x 3 warmups +
# 1 capture attempt) and a permanent eager fallback for all 1151 calls.
R4_DECLINED_PROFILE = {
    "timing_schema_version": 1,
    "synchronized": True,
    "phases": {
        "dispatch.candidate_total": {"calls": 1151, "mean_ms": 17.5434, "total_seconds": 20.192496},
        "dispatch.native_fallback": {"calls": 546, "mean_ms": 1.6457, "total_seconds": 0.898562},
        "dispatch.native_reference": {"calls": 64, "mean_ms": 2.6076, "total_seconds": 0.166885},
        "dispatch.shape_key": {"calls": 1761, "mean_ms": 0.1096, "total_seconds": 0.193015},
        "shadow.native_forward": {"calls": 1151, "mean_ms": 8.0018, "total_seconds": 9.210115},
        "subgraph.execute": {"calls": 1151, "mean_ms": 10.7404, "total_seconds": 12.362185},
        "subgraph.execute_cuda_graph": {"calls": 192, "mean_ms": 0.0125, "total_seconds": 0.002407},
        "subgraph.flatten": {"calls": 1151, "mean_ms": 0.0659, "total_seconds": 0.07589},
        "subgraph.unflatten": {"calls": 1151, "mean_ms": 0.0466, "total_seconds": 0.053685},
        "subgraph.validate": {"calls": 1151, "mean_ms": 0.0451, "total_seconds": 0.051853},
    },
    "notes": {},
}

# A healthy CUDA-graph profile: 16 generations over 48 scopes = 6144 candidate
# calls; 144 warmups + 48 captures, the rest replays.
HEALTHY_GRAPH_PROFILE = {
    "timing_schema_version": 1,
    "synchronized": True,
    "phases": {
        "dispatch.candidate_total": {"calls": 6144, "mean_ms": 7.9, "total_seconds": 48.5376},
        "dispatch.shape_key": {"calls": 6250, "mean_ms": 0.1, "total_seconds": 0.625},
        "shadow.native_forward": {"calls": 6144, "mean_ms": 8.2, "total_seconds": 50.3808},
        "subgraph.execute": {"calls": 192, "mean_ms": 10.0, "total_seconds": 1.92},
        "subgraph.execute_cuda_graph": {"calls": 6144, "mean_ms": 0.35, "total_seconds": 2.1504},
        "subgraph.flatten": {"calls": 6144, "mean_ms": 0.06, "total_seconds": 0.36864},
        "subgraph.unflatten": {"calls": 6144, "mean_ms": 0.045, "total_seconds": 0.27648},
        "subgraph.validate": {"calls": 6144, "mean_ms": 0.045, "total_seconds": 0.27648},
    },
    "notes": {"cuda_graph_warmup": 144},
}


class TimingReportValidationTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        report = TimingReport.from_dict(R4_EAGER_PROFILE)
        self.assertTrue(report.synchronized)
        self.assertEqual(report.phase("shadow.native_forward").calls, 1151)
        self.assertAlmostEqual(report.phase("shadow.native_forward").mean_ms, 8.1758, places=3)

    def test_rejects_bad_version(self) -> None:
        payload = dict(R4_EAGER_PROFILE, timing_schema_version=2)
        with self.assertRaises(DispatchAnalysisError):
            TimingReport.from_dict(payload)

    def test_rejects_unsynchronized_flag_type(self) -> None:
        payload = dict(R4_EAGER_PROFILE, synchronized="yes")
        with self.assertRaises(DispatchAnalysisError):
            TimingReport.from_dict(payload)

    def test_rejects_zero_calls_with_nonzero_total(self) -> None:
        payload = json.loads(json.dumps(R4_EAGER_PROFILE))
        payload["phases"]["subgraph.execute"] = {"calls": 0, "total_seconds": 1.0}
        with self.assertRaises(DispatchAnalysisError):
            TimingReport.from_dict(payload)

    def test_declined_capture_notes_parsed(self) -> None:
        payload = json.loads(json.dumps(R4_DECLINED_PROFILE))
        payload["notes"] = {
            "cuda_graph_warmup": 144,
            "cuda_graph_declined: runtime input 10 is bool, not a tensor": 48,
        }
        report = TimingReport.from_dict(payload)
        self.assertEqual(report.note_count("cuda_graph_warmup"), 144)
        self.assertEqual(
            report.declined_captures(),
            {"runtime input 10 is bool, not a tensor": 48},
        )


class AttributionTest(unittest.TestCase):
    def test_r4_eager_replay_penalty_reproduced(self) -> None:
        """The R4 section 7 attribution: 11.57 ms replay vs 8.18 ms native."""
        attribution = attribute_overhead(TimingReport.from_dict(R4_EAGER_PROFILE))
        self.assertEqual(attribution.replay_path, "eager")
        self.assertAlmostEqual(attribution.replay_mean_ms, 11.5666, places=3)
        self.assertAlmostEqual(attribution.native_forward_mean_ms, 8.1758, places=3)
        self.assertAlmostEqual(attribution.net_overhead_ms_per_call, 18.4899 - 8.1758, places=3)
        # Plumbing: flatten + validate + unflatten amortized over candidate calls.
        expected_plumbing = (0.07265 + 0.051488 + 0.050657) * 1000.0 / 1151
        self.assertAlmostEqual(attribution.plumbing_ms_per_candidate_call, expected_plumbing, places=4)
        self.assertLess(attribution.plumbing_ms_per_candidate_call, 0.2)

    def test_declined_capture_profile_is_mixed(self) -> None:
        attribution = attribute_overhead(TimingReport.from_dict(R4_DECLINED_PROFILE))
        self.assertEqual(attribution.replay_path, "mixed")
        self.assertEqual(attribution.graph_replay_calls, 192)
        self.assertEqual(attribution.eager_execute_calls, 1151)

    def test_healthy_graph_profile(self) -> None:
        attribution = attribute_overhead(TimingReport.from_dict(HEALTHY_GRAPH_PROFILE))
        self.assertEqual(attribution.replay_path, "mixed")
        self.assertEqual(attribution.warmup_calls, 144)
        # Candidate path is faster than native: negative overhead.
        self.assertLess(attribution.net_overhead_ms_per_call, 0.0)

    def test_pure_graph_profile(self) -> None:
        payload = json.loads(json.dumps(HEALTHY_GRAPH_PROFILE))
        del payload["phases"]["subgraph.execute"]
        attribution = attribute_overhead(TimingReport.from_dict(payload))
        self.assertEqual(attribution.replay_path, "cuda_graph")
        self.assertAlmostEqual(attribution.replay_mean_ms, 0.35, places=3)

    def test_requires_shadow_mode(self) -> None:
        payload = dict(R4_EAGER_PROFILE, synchronized=False)
        with self.assertRaises(DispatchAnalysisError):
            attribute_overhead(TimingReport.from_dict(payload))

    def test_requires_matching_shadow_counts(self) -> None:
        payload = json.loads(json.dumps(R4_EAGER_PROFILE))
        payload["phases"]["shadow.native_forward"] = {"calls": 100, "total_seconds": 0.8}
        with self.assertRaises(DispatchAnalysisError):
            attribute_overhead(TimingReport.from_dict(payload))

    def test_requires_candidate_calls(self) -> None:
        payload = json.loads(json.dumps(R4_EAGER_PROFILE))
        del payload["phases"]["dispatch.candidate_total"]
        with self.assertRaises(DispatchAnalysisError):
            attribute_overhead(TimingReport.from_dict(payload))


class E2EOverheadTest(unittest.TestCase):
    def test_reproduces_3104(self) -> None:
        """The R4 arithmetic: 3.104 ms/call from the eager FX replay era."""
        result = overhead_from_e2e(
            native_median_seconds=3.6789,
            candidate_median_seconds=4.8226,
            calls_per_generation=384,
            kernel_saving_ms_per_call=0.124,
        )
        self.assertAlmostEqual(result.net_cost_ms_per_call, 2.980, places=2)
        self.assertAlmostEqual(result.overhead_ms_per_call, 3.104, places=2)

    def test_negative_overhead_when_framework_accelerates(self) -> None:
        """Gate-5-era arithmetic: candidate faster than native end-to-end."""
        result = overhead_from_e2e(
            native_median_seconds=3.7494,
            candidate_median_seconds=2.9963,
            calls_per_generation=384,
            kernel_saving_ms_per_call=0.124,
        )
        self.assertLess(result.net_cost_ms_per_call, 0.0)
        self.assertLess(result.overhead_ms_per_call, 0.0)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(DispatchAnalysisError):
            overhead_from_e2e(
                native_median_seconds=3.0,
                candidate_median_seconds=3.0,
                calls_per_generation=0,
                kernel_saving_ms_per_call=0.1,
            )


class BreakEvenTest(unittest.TestCase):
    def test_r4_verdict_25x_short(self) -> None:
        """At 3.104 ms overhead and 384 calls/generation, the 124 us kernel
        was ~25x short of clearing the 1.01x gate."""
        required = required_saving_ms_per_call(
            native_e2e_seconds=3.6789,
            calls_per_generation=384,
            overhead_ms_per_call=3.104,
            gate=1.01,
        )
        self.assertAlmostEqual(required, 3.104 + 0.0949, places=3)
        self.assertAlmostEqual(required / 0.124, 25.8, places=0)

    def test_gate_one_reduces_to_overhead(self) -> None:
        required = required_saving_ms_per_call(
            native_e2e_seconds=3.6789,
            calls_per_generation=384,
            overhead_ms_per_call=0.05,
            gate=1.0,
        )
        self.assertAlmostEqual(required, 0.05, places=6)

    def test_low_volume_regions_need_little(self) -> None:
        """VAE-scale call volumes: even the old 3.104 ms tax was affordable
        for a large enough kernel; the gate term dominates at low volume."""
        required = required_saving_ms_per_call(
            native_e2e_seconds=3.6789,
            calls_per_generation=5,
            overhead_ms_per_call=0.05,
            gate=1.01,
        )
        # 0.05 + 3.6789 * 0.0099010 * 1000 / 5 = 7.34 ms
        self.assertAlmostEqual(required, 7.337, places=2)

    def test_curve_spans_grid(self) -> None:
        curve = breakeven_curve(
            native_e2e_seconds=3.36, overhead_ms_per_call=0.05, gate=1.01
        )
        self.assertEqual(len(curve), 12)
        self.assertEqual(curve[0].calls_per_generation, 1)
        # Monotonically decreasing in call volume.
        required = [point.required_saving_ms_per_call for point in curve]
        self.assertEqual(required, sorted(required, reverse=True))
        # Floor is the overhead itself at infinite volume.
        self.assertGreater(curve[-1].required_saving_ms_per_call, 0.05)

    def test_rejects_invalid(self) -> None:
        with self.assertRaises(DispatchAnalysisError):
            required_saving_ms_per_call(
                native_e2e_seconds=-1.0,
                calls_per_generation=10,
                overhead_ms_per_call=0.0,
            )
        with self.assertRaises(DispatchAnalysisError):
            required_saving_ms_per_call(
                native_e2e_seconds=3.0,
                calls_per_generation=10,
                overhead_ms_per_call=0.0,
                gate=0.5,
            )


if __name__ == "__main__":
    unittest.main()


class HostProfileSummaryTest(unittest.TestCase):
    def test_host_only_profile(self) -> None:
        from autokernel.dispatch import host_profile_summary

        payload = {
            "timing_schema_version": 1,
            "synchronized": False,
            "phases": {
                "dispatch.candidate_total": {"calls": 5952, "mean_ms": 0.31, "total_seconds": 1.84512},
                "dispatch.shape_key": {"calls": 5952, "mean_ms": 0.11, "total_seconds": 0.65472},
                "subgraph.execute_cuda_graph": {"calls": 5952, "mean_ms": 0.02, "total_seconds": 0.11904},
                "subgraph.flatten": {"calls": 5952, "mean_ms": 0.07, "total_seconds": 0.41664},
                "subgraph.unflatten": {"calls": 5952, "mean_ms": 0.05, "total_seconds": 0.2976},
                "subgraph.validate": {"calls": 5952, "mean_ms": 0.05, "total_seconds": 0.2976},
            },
            "notes": {"cuda_graph_warmup": 144},
        }
        summary = host_profile_summary(TimingReport.from_dict(payload))
        self.assertAlmostEqual(summary["candidate_total_host_ms_per_call"], 0.31, places=4)
        self.assertAlmostEqual(summary["graph_replay_host_ms_per_call"], 0.02, places=4)
        self.assertAlmostEqual(
            summary["plumbing_host_ms_per_call"],
            (0.41664 + 0.2976 + 0.2976) * 1000.0 / 5952,
            places=4,
        )
        self.assertEqual(summary["warmup_calls"], 144)

    def test_rejects_synchronized_report(self) -> None:
        from autokernel.dispatch import host_profile_summary

        with self.assertRaises(DispatchAnalysisError):
            host_profile_summary(TimingReport.from_dict(R4_EAGER_PROFILE))
