"""Bounded energy-memory runtime used by the CLBench adapter.

The BSM implementation deliberately keeps the task-specific code at the edge:
prompt parsing and schema decoding live in ``system.py``; the persistent state
here is a bounded energy landscape over candidate transmitter basins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


EPS = 1e-9


@dataclass
class SpectrumPeak:
    """A visible detector peak parsed from one BSM scan."""

    center_freq: float
    bandwidth: float
    power_dbm: float
    peak_id: str = ""


@dataclass
class BandInfo:
    """Current observed band metadata."""

    start_mhz: float = 0.0
    end_mhz: float = 180.0
    noise_floor_dbm: float = -50.0


@dataclass
class EnergyBasin:
    """One bounded basin in the energy landscape."""

    center_freq: float
    bandwidth: float
    power_dbm: float
    logit: float
    hits: int
    created_at: int
    last_seen: int
    active_logit: float = 0.0
    miss_count: int = 0


@dataclass
class SpectrumEnergyConfig:
    """Hyperparameters for the BSM energy-memory runtime."""

    capacity: int = 32
    merge_radius_mhz: float = 4.0
    bandwidth_merge_factor: float = 0.75
    write_lr: float = 1.15
    positive_width_floor_mhz: float = 3.5
    current_power_margin_db: float = 9.0
    existence_decay: float = 0.995
    active_decay: float = 0.45
    age_energy_weight: float = 0.035
    min_report_hits: int = 2
    report_logit_threshold: float = 0.65
    current_report_logit_threshold: float = 0.20
    max_report: int = 24
    sliding_window: int = 8
    logit_radius: float = 8.0
    min_bandwidth_mhz: float = 1.0
    max_bandwidth_mhz: float = 30.0


def sigmoid(x: float) -> float:
    """Numerically stable scalar logistic."""

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


class SpectrumEnergyMemoryRuntime:
    """Bounded online energy memory for Blind Spectrum Monitoring.

    The persistent state is a fixed-capacity list of energy basins.  Each basin
    stores a candidate transmitter location plus a bounded natural parameter
    ``logit``.  Positive scan evidence applies the logistic-loss gradient

        logit <- logit - alpha * (sigmoid(logit) - 1),

    lowering the basin's energy.  Propagation applies slow relaxation and
    active-state decay without erasing dormant transmitter hypotheses.
    """

    def __init__(
        self,
        cfg: SpectrumEnergyConfig | None = None,
        *,
        write_enabled: bool = True,
        propagation_enabled: bool = True,
        sliding_window_enabled: bool = False,
    ) -> None:
        self.cfg = cfg or SpectrumEnergyConfig()
        self.write_enabled = write_enabled
        self.propagation_enabled = propagation_enabled
        self.sliding_window_enabled = sliding_window_enabled
        self.step_index = 0
        self.basins: list[EnergyBasin] = []
        self.last_update: dict[str, Any] = {}

    def reset(self) -> None:
        self.step_index = 0
        self.basins = []
        self.last_update = {}

    def update_from_scan(
        self,
        peaks: list[SpectrumPeak],
        *,
        band: BandInfo,
        scan_index: int | None = None,
    ) -> None:
        """Update the bounded energy state from one visible scan."""

        self.step_index = self.step_index + 1 if scan_index is None else scan_index
        for basin in self.basins:
            basin.active_logit *= self.cfg.active_decay
            basin.miss_count += 1
            if self.propagation_enabled:
                basin.logit *= self.cfg.existence_decay

        matched_basin_ids: set[int] = set()
        matched_peak_ids: set[int] = set()

        for peak_idx, peak in enumerate(peaks):
            basin_idx = self._nearest_basin_index(peak, exclude=matched_basin_ids)
            if basin_idx is None:
                continue
            self._absorb_peak(self.basins[basin_idx], peak, band)
            matched_basin_ids.add(basin_idx)
            matched_peak_ids.add(peak_idx)

        for peak_idx, peak in enumerate(peaks):
            if peak_idx in matched_peak_ids:
                continue
            self._allocate_basin(peak, band)

        if self.sliding_window_enabled:
            horizon = max(1, int(self.cfg.sliding_window))
            self.basins = [
                basin
                for basin in self.basins
                if self.step_index - basin.last_seen <= horizon
            ]

        self.last_update = {
            "scan_index": self.step_index,
            "peaks": len(peaks),
            "matched": len(matched_peak_ids),
            "basins": len(self.basins),
            "write_enabled": self.write_enabled,
            "propagation_enabled": self.propagation_enabled,
            "sliding_window_enabled": self.sliding_window_enabled,
        }

    def reports(self, *, band: BandInfo) -> list[dict[str, Any]]:
        """Decode low-energy basins into CLBench transmitter dictionaries."""

        candidates: list[tuple[float, EnergyBasin]] = []
        for basin in self.basins:
            if not self._is_reportable(basin, band):
                continue
            candidates.append((self.energy(basin), basin))

        candidates.sort(key=lambda item: item[0])
        reports: list[dict[str, Any]] = []
        for _, basin in candidates[: self.cfg.max_report]:
            active_prob = sigmoid(basin.active_logit)
            reports.append(
                {
                    "center_freq": round(
                        _clip(basin.center_freq, band.start_mhz, band.end_mhz), 2
                    ),
                    "bandwidth": round(
                        _clip(
                            basin.bandwidth,
                            self.cfg.min_bandwidth_mhz,
                            self.cfg.max_bandwidth_mhz,
                        ),
                        2,
                    ),
                    "currently_active": active_prob >= 0.5,
                    "estimated_power": round(basin.power_dbm, 2),
                }
            )
        return reports

    def vanilla_online_reports(self, *, band: BandInfo) -> list[dict[str, Any]]:
        """Decode persistent candidates without using energy/logit state.

        This attribution baseline keeps the same parser, candidate matching,
        bounded capacity, EMA geometry, and report schema as ``reports``. It
        removes the energy-specific parts: no existence logit, no logistic
        write, and no energy sorting.
        """

        candidates: list[tuple[tuple[float, float, float], EnergyBasin]] = []
        for basin in self.basins:
            if not self._is_vanilla_reportable(basin, band):
                continue
            age = max(0, self.step_index - basin.last_seen)
            sort_key = (-float(basin.hits), float(age), -basin.power_dbm)
            candidates.append((sort_key, basin))

        candidates.sort(key=lambda item: item[0])
        reports: list[dict[str, Any]] = []
        for _, basin in candidates[: self.cfg.max_report]:
            active_prob = sigmoid(basin.active_logit)
            reports.append(
                {
                    "center_freq": round(
                        _clip(basin.center_freq, band.start_mhz, band.end_mhz), 2
                    ),
                    "bandwidth": round(
                        _clip(
                            basin.bandwidth,
                            self.cfg.min_bandwidth_mhz,
                            self.cfg.max_bandwidth_mhz,
                        ),
                        2,
                    ),
                    "currently_active": active_prob >= 0.5,
                    "estimated_power": round(basin.power_dbm, 2),
                }
            )
        return reports

    def current_scan_reports(
        self, peaks: list[SpectrumPeak], *, band: BandInfo
    ) -> list[dict[str, Any]]:
        """A non-learning current-scan baseline using the same parser/decoder."""

        reports = []
        for peak in peaks[: self.cfg.max_report]:
            reports.append(
                {
                    "center_freq": round(
                        _clip(peak.center_freq, band.start_mhz, band.end_mhz), 2
                    ),
                    "bandwidth": round(
                        _clip(
                            peak.bandwidth,
                            self.cfg.min_bandwidth_mhz,
                            self.cfg.max_bandwidth_mhz,
                        ),
                        2,
                    ),
                    "currently_active": True,
                    "estimated_power": round(peak.power_dbm, 2),
                }
            )
        return reports

    def energy(self, basin: EnergyBasin) -> float:
        age = max(0, self.step_index - basin.last_seen)
        return -basin.logit + self.cfg.age_energy_weight * math.log1p(age)

    def artifacts(self) -> dict[str, Any]:
        return {
            "runtime": "spectrum_energy_memory",
            "config": asdict(self.cfg),
            "step_index": self.step_index,
            "basin_count": len(self.basins),
            "last_update": dict(self.last_update),
            "basins": [
                {
                    **asdict(basin),
                    "energy": round(self.energy(basin), 6),
                    "existence_prob": round(sigmoid(basin.logit), 6),
                    "active_prob": round(sigmoid(basin.active_logit), 6),
                }
                for basin in sorted(self.basins, key=self.energy)
            ],
        }

    def _nearest_basin_index(
        self, peak: SpectrumPeak, *, exclude: set[int]
    ) -> int | None:
        best_idx = None
        best_score = float("inf")
        for idx, basin in enumerate(self.basins):
            if idx in exclude:
                continue
            center_dist = abs(peak.center_freq - basin.center_freq)
            if center_dist > self.cfg.merge_radius_mhz:
                continue
            bw_scale = max(peak.bandwidth, basin.bandwidth, 1.0)
            bw_dist = abs(peak.bandwidth - basin.bandwidth) / bw_scale
            if bw_dist > self.cfg.bandwidth_merge_factor:
                continue
            score = center_dist + 1.5 * bw_dist
            if score < best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _allocate_basin(self, peak: SpectrumPeak, band: BandInfo) -> None:
        if len(self.basins) >= self.cfg.capacity:
            replace_idx = self._weakest_basin_index()
            if replace_idx is None:
                return
            self.basins.pop(replace_idx)

        initial_logit = -0.15
        basin = EnergyBasin(
            center_freq=_clip(peak.center_freq, band.start_mhz, band.end_mhz),
            bandwidth=_clip(
                peak.bandwidth, self.cfg.min_bandwidth_mhz, self.cfg.max_bandwidth_mhz
            ),
            power_dbm=peak.power_dbm,
            logit=initial_logit,
            hits=0,
            created_at=self.step_index,
            last_seen=self.step_index,
        )
        self._absorb_peak(basin, peak, band)
        self.basins.append(basin)

    def _absorb_peak(
        self, basin: EnergyBasin, peak: SpectrumPeak, band: BandInfo
    ) -> None:
        basin.hits += 1
        basin.miss_count = 0
        basin.last_seen = self.step_index

        eta = 1.0 / max(float(basin.hits), 1.0)
        basin.center_freq += eta * (peak.center_freq - basin.center_freq)
        basin.bandwidth += eta * (peak.bandwidth - basin.bandwidth)
        basin.bandwidth = _clip(
            basin.bandwidth, self.cfg.min_bandwidth_mhz, self.cfg.max_bandwidth_mhz
        )
        basin.power_dbm += eta * (peak.power_dbm - basin.power_dbm)
        basin.active_logit = _clip(basin.active_logit + 1.75, -self.cfg.logit_radius, self.cfg.logit_radius)

        if not self.write_enabled:
            return

        strength = self._evidence_strength(peak, band)
        p = sigmoid(basin.logit)
        # Gradient step on BCE(label=1) for the basin's existence logit.
        basin.logit += self.cfg.write_lr * strength * (1.0 - p)
        basin.logit = _clip(basin.logit, -self.cfg.logit_radius, self.cfg.logit_radius)

    def _evidence_strength(self, peak: SpectrumPeak, band: BandInfo) -> float:
        width_score = _clip(
            (peak.bandwidth - self.cfg.positive_width_floor_mhz) / 10.0,
            0.15,
            1.0,
        )
        power_margin = peak.power_dbm - band.noise_floor_dbm
        power_score = _clip(power_margin / 18.0, 0.15, 1.0)
        return 0.5 + 0.5 * min(width_score, power_score)

    def _is_reportable(self, basin: EnergyBasin, band: BandInfo) -> bool:
        active_now = self.step_index == basin.last_seen
        power_margin = basin.power_dbm - band.noise_floor_dbm
        current_credible = (
            active_now
            and basin.bandwidth >= self.cfg.positive_width_floor_mhz
            and power_margin >= self.cfg.current_power_margin_db
            and basin.logit >= self.cfg.current_report_logit_threshold
        )
        persistent_credible = (
            basin.hits >= self.cfg.min_report_hits
            and basin.logit >= self.cfg.report_logit_threshold
        )
        return current_credible or persistent_credible

    def _is_vanilla_reportable(self, basin: EnergyBasin, band: BandInfo) -> bool:
        active_now = self.step_index == basin.last_seen
        power_margin = basin.power_dbm - band.noise_floor_dbm
        peak_quality = (
            basin.bandwidth >= self.cfg.positive_width_floor_mhz
            and power_margin >= self.cfg.current_power_margin_db
        )
        current_credible = active_now and peak_quality
        persistent_credible = basin.hits >= self.cfg.min_report_hits and peak_quality
        return current_credible or persistent_credible

    def _weakest_basin_index(self) -> int | None:
        if not self.basins:
            return None
        weakest_idx = 0
        weakest_score = -float("inf")
        for idx, basin in enumerate(self.basins):
            age = max(0, self.step_index - basin.last_seen)
            score = self.energy(basin) + 0.05 * age - 0.15 * basin.hits
            if score > weakest_score:
                weakest_score = score
                weakest_idx = idx
        return weakest_idx
