#!/usr/bin/env python3
"""Compute frozen held-out routing coverage for B1-CWE and CAA-CWE."""

from __future__ import annotations

from collections import Counter

from phase16_analysis_common import OUT, load_all, write_csv


def main() -> int:
    loaded = load_all()
    rows = []
    for method in ("B1-CWE", "CAA-CWE"):
        generation = loaded["data"][(method, 42)]["generation"]
        routes = Counter(str(row.get("route_status")) for row in generation)
        if method == "B1-CWE":
            supported = routes["ROUTE_GUIDANCE_UNAVAILABLE"]
            unsupported = routes["UNSUPPORTED_ROUTE_FALLBACK_B1"]
            active = sum(row.get("guidance_available") is True for row in generation)
            fallback = sum(row.get("fallback_status") == "SUBMITTED_B1" for row in generation)
            no_intervention = 0
            reason = "No route-specific guidance was available; every record used the B1 fallback"
        else:
            supported = routes["SUPPORTED_VECTOR_APPLIED"]
            unsupported = routes["UNSUPPORTED_ROUTE_NO_INTERVENTION"]
            active = sum(row.get("intervention_applied") is True for row in generation)
            fallback = 0
            no_intervention = sum(row.get("intervention_applied") is False for row in generation)
            reason = "Frozen supported vectors applied; unsupported routes use explicit no-intervention"
        rows.append({
            "method": method, "seed": 42, "seed_status": "SINGLE_SEED", "total_prompts": len(generation),
            "supported_route_prompts": supported, "unsupported_route_prompts": unsupported,
            "active_guidance_or_intervention_prompts": active,
            "fallback_prompts": fallback, "no_intervention_prompts": no_intervention,
            "active_coverage_rate": f"{active / len(generation):.6f}",
            "route_status_counts": str(dict(sorted(routes.items()))), "reason": reason,
        })
    write_csv(OUT / "phase16_routing_coverage.csv", list(rows[0]), rows)
    print("COMPLETE: phase16_routing_coverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
