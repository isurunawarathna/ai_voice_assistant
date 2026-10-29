"""Dependency-free metrics in the Prometheus text exposition format.

The onboarding funnel is the first thing worth watching in production:
how many callers connect, how many pick a language, how, and how long it
takes. A few counters and one histogram answer that without pulling in a
client library. Everything runs on the event loop thread, but a lock keeps
the registry safe if it is ever touched from a worker thread.
"""

from __future__ import annotations

import threading
from collections import defaultdict

Labels = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS: tuple[float, ...] = (1, 2, 5, 10, 20, 30, 60, 120, 300)


def _labels(values: dict[str, str]) -> Labels:
    return tuple(sorted(values.items()))


def _render_labels(labels: Labels, extra: tuple[tuple[str, str], ...] = ()) -> str:
    pairs = (*labels, *extra)
    if not pairs:
        return ""
    body = ",".join(f'{key}="{_escape(value)}"' for key, value in pairs)
    return "{" + body + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Counter:
    def __init__(self, name: str, help_text: str, lock: threading.Lock) -> None:
        self.name = name
        self.help = help_text
        self._lock = lock
        self._values: dict[Labels, float] = defaultdict(float)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._values[_labels(labels)] += amount

    def value(self, **labels: str) -> float:
        with self._lock:
            return self._values.get(_labels(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            for labels, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_render_labels(labels)} {value:g}")
        return lines


class Gauge:
    def __init__(self, name: str, help_text: str, lock: threading.Lock) -> None:
        self.name = name
        self.help = help_text
        self._lock = lock
        self._value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def render(self) -> list[str]:
        return [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} gauge",
            f"{self.name} {self.value:g}",
        ]


class Histogram:
    def __init__(
        self,
        name: str,
        help_text: str,
        lock: threading.Lock,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> None:
        self.name = name
        self.help = help_text
        self._lock = lock
        self._buckets = tuple(sorted(buckets))
        self._counts: dict[Labels, list[int]] = {}
        self._sums: dict[Labels, float] = defaultdict(float)
        self._totals: dict[Labels, int] = defaultdict(int)

    def observe(self, value: float, **labels: str) -> None:
        key = _labels(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * len(self._buckets))
            for index, bound in enumerate(self._buckets):
                if value <= bound:
                    counts[index] += 1
            self._sums[key] += value
            self._totals[key] += 1

    def count(self, **labels: str) -> int:
        with self._lock:
            return self._totals.get(_labels(labels), 0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key in sorted(self._counts):
                for bound, count in zip(self._buckets, self._counts[key], strict=True):
                    lines.append(
                        f"{self.name}_bucket"
                        f"{_render_labels(key, (('le', f'{bound:g}'),))} {count}"
                    )
                lines.append(
                    f"{self.name}_bucket"
                    f"{_render_labels(key, (('le', '+Inf'),))} {self._totals[key]}"
                )
                lines.append(f"{self.name}_sum{_render_labels(key)} {self._sums[key]:g}")
                lines.append(
                    f"{self.name}_count{_render_labels(key)} {self._totals[key]}"
                )
        return lines


class Metrics:
    """Every metric the service exposes, created once per container."""

    def __init__(self) -> None:
        lock = threading.Lock()

        self.connections = Counter(
            "voice_connections_total",
            "WebSocket connections by result (accepted, rejected_origin, "
            "rejected_rate_limit, rejected_capacity).",
            lock,
        )
        self.active_conversations = Gauge(
            "voice_active_conversations", "Conversations currently open.", lock
        )
        self.prompts_played = Counter(
            "voice_prompts_played_total", "Pre-recorded prompts sent.", lock
        )
        self.language_attempts = Counter(
            "voice_language_attempts_total",
            "Language selection attempts that did not resolve, by reason.",
            lock,
        )
        self.language_selected = Counter(
            "voice_language_selected_total",
            "Languages chosen, by language and method.",
            lock,
        )
        self.onboarding_outcomes = Counter(
            "voice_onboarding_outcomes_total",
            "How the welcome and language flow ended "
            "(selected, failed, timeout, abandoned, too_many_invalid_inputs, "
            "too_many_resets).",
            lock,
        )
        self.invalid_inputs = Counter(
            "voice_invalid_inputs_total",
            "Unusable client inputs, by kind.",
            lock,
        )
        self.selection_seconds = Histogram(
            "voice_language_selection_seconds",
            "Seconds from the end of the welcome prompt to a chosen language.",
            lock,
        )
        self._all = (
            self.connections,
            self.active_conversations,
            self.prompts_played,
            self.language_attempts,
            self.language_selected,
            self.onboarding_outcomes,
            self.invalid_inputs,
            self.selection_seconds,
        )

    def render(self) -> str:
        lines: list[str] = []
        for metric in self._all:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"
