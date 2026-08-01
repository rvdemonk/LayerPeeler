"""Per-stage wall-clock, as a first-class output.

This is not logging garnish. The number this module produces is the input
to SLA design and unit-economics modelling: a customer-facing promise
("upload a mascot, get N emotes by X") can only be made from measured
stage times, and the split between the purchased stage (fal generation,
which scales with spend) and the local stages (which scale with our CPU)
is what decides whether more throughput costs dollars or machines.

So: every stage is timed, the record is machine-readable, and it is
written even when the run fails — a run that died in matting still tells
you what generation cost in seconds.

Deliberately dependency-free and process-local. Wall clock, not CPU time,
because wall clock is what the customer waits.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path


class Timings:
    """Ordered stage record for one pipeline run.

    Stages may repeat (encode runs once per profile); each entry is kept
    separately rather than summed, so a slow rung is visible instead of
    hidden inside a category total.
    """

    def __init__(self, run_name, meta=None):
        self.run = run_name
        self.meta = dict(meta or {})
        self.stages = []
        self.t_start = time.time()

    @contextmanager
    def stage(self, name, **info):
        """Time a stage. Records even if the body raises, then re-raises.

        The `error` field distinguishes "this stage took 4s" from "this
        stage took 4s and then blew up" — without it a failed run's
        timings read as a fast run's timings.
        """
        t0 = time.time()
        rec = {"stage": name, "seconds": None, **info}
        self.stages.append(rec)
        try:
            yield rec
        except BaseException as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
            raise
        finally:
            rec["seconds"] = round(time.time() - t0, 3)

    def add(self, name, seconds, **info):
        """Record a stage timed elsewhere (e.g. a remote queue's own clock)."""
        self.stages.append({"stage": name, "seconds": round(seconds, 3), **info})

    def total(self):
        return round(time.time() - self.t_start, 3)

    def seconds_for(self, *names):
        return round(sum(s["seconds"] or 0 for s in self.stages
                         if s["stage"] in names), 3)

    def record(self):
        """The machine-readable artifact.

        `generation_seconds` / `local_seconds` are split out at the top
        level because that is the ratio business modelling actually
        consumes: generation is bought per clip and is latency we cannot
        engineer away; local is CPU we own and can parallelise.
        """
        total = self.total()
        gen = self.seconds_for("generate", "download")
        return {
            "run": self.run,
            "meta": self.meta,
            "stages": self.stages,
            "total_seconds": total,
            "generation_seconds": gen,
            "local_seconds": round(total - gen, 3),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record(), indent=1))
        return path

    def summary(self):
        """Human table. Same numbers as record(), no rounding surprises."""
        rec = self.record()
        w = max([len(s["stage"]) for s in self.stages] + [12])
        lines = ["  %-*s %8s  %s" % (w, "stage", "sec", "detail")]
        for s in self.stages:
            detail = " ".join(
                "%s=%s" % (k, v) for k, v in s.items()
                if k not in ("stage", "seconds") and v is not None)
            lines.append("  %-*s %8.2f  %s"
                         % (w, s["stage"], s["seconds"] or 0.0, detail))
        lines.append("  %-*s %8.2f" % (w, "TOTAL", rec["total_seconds"]))
        lines.append("  %-*s %8.2f  (%.2f local)"
                     % (w, "of which gen", rec["generation_seconds"],
                        rec["local_seconds"]))
        return "\n".join(lines)
