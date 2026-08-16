import numpy as np
from dataclasses import dataclass, field


@dataclass
class LatencyTracker:
    records: list[dict] = field(default_factory=list)

    def record(self, latencies: dict):
        self.records.append(latencies)

    def report(self) -> dict:
        if not self.records:
            return {}
        stages = set()
        for r in self.records:
            stages.update(r.keys())
        result = {}
        for stage in sorted(stages):
            values = sorted(r[stage] for r in self.records if stage in r)
            if not values:
                continue
            result[stage] = {
                "p50": float(np.percentile(values, 50)),
                "p70": float(np.percentile(values, 70)),
                "p100": float(np.max(values)),
                "mean": float(np.mean(values)),
                "count": len(values),
            }
        return result

    def format_report(self) -> str:
        report = self.report()
        if not report:
            return "No queries recorded yet."
        lines = [f"{'Stage':<30} {'P50':>8} {'P70':>8} {'P100':>8}"]
        lines.append("-" * 60)
        for stage, s in report.items():
            lines.append(f"{stage:<30} {s['p50']:>7.1f}ms {s['p70']:>7.1f}ms {s['p100']:>7.1f}ms")
        lines.append(f"\nTotal queries: {len(self.records)}")
        return "\n".join(lines)
