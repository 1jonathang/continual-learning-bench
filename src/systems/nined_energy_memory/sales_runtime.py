"""Public-only Sales Prediction runtime for ``nined_energy_memory``.

The runtime consumes only evidence visible to the agent:

* prompt metadata and required forecast entries;
* CSV/JSON files under ``/app/data`` read by a bash command;
* previous-round feedback embedded in later prompts.

It deliberately exposes several attribution modes.  The important comparison is
not whether a hand-built forecaster can score well, but whether persistent
online state beats current-only/static controls, and whether the energy-style
readout adds anything over transparent full-horizon accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any


_SUMMARY_MARKER = "NINED_SALES_SUMMARY_JSON="

_FURNITURE_ID_KEYS = ("furniture_id", "product_id", "item_code")
_LOCATION_ID_KEYS = ("location_id", "store_id", "branch_id")
_YEAR_KEYS = ("year", "sale_year", "txn_year", "date")
_ITEMS_KEYS = ("items_sold", "quantity", "units_sold")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def _semantic_cluster(type_id: str, name: str = "") -> str:
    """Domain-level cluster heuristic from public type/name text."""
    token = f"{type_id} {name}".upper()
    if "DESK" in token or "WORK" in token:
        return "workspace"
    if "SOFA" in token or "CHAIR" in token:
        return "seating"
    if "TABLE" in token:
        return "tables"
    if "BED" in token or "NIGHTSTAND" in token:
        return "bedroom"
    if "BOOKCASE" in token or "DRESSER" in token or "CABINET" in token or "SIDEBOARD" in token:
        return "storage"
    return re.sub(r"[^a-z0-9]+", "_", f"{type_id}".lower()).strip("_") or "unknown"


@dataclass(frozen=True)
class RequiredSalesEntry:
    locality: str
    furniture_name: str
    year: int


@dataclass(frozen=True)
class SalesContext:
    instance_idx: int
    active_instance_idx: int
    target_year: int
    forecast_years: tuple[int, ...]
    step: str
    required: tuple[RequiredSalesEntry, ...]
    feedback_rows: tuple[dict[str, Any], ...] = ()


@dataclass
class SalesRecord:
    locality: str
    location_id: str
    furniture_name: str
    furniture_id: str
    furniture_type: str
    cluster: str
    price: float
    year: int
    items_sold: float
    source: str
    instance_idx: int

    def key(self) -> tuple[str, str, int]:
        return (self.locality, self.furniture_name, self.year)

    def entity(self) -> tuple[str, str]:
        return (self.locality, self.furniture_name)


@dataclass
class SalesMemoryConfig:
    sliding_window: int = 4
    max_records: int = 4096
    min_growth: float = math.log(1.02)
    max_growth: float = math.log(1.45)
    default_growth: float = math.log(1.16)
    price_scale: float = 1000.0
    demand_floor: float = 100.0


@dataclass
class SalesReadout:
    mode: str
    read_policy: str
    record_count: int
    current_record_count: int
    feedback_record_count: int
    growth_by_cluster: dict[str, float]
    source_counts: dict[str, int] = field(default_factory=dict)


class SalesMemoryRuntime:
    """Stateful public-data sales forecaster with attribution modes."""

    def __init__(self, cfg: SalesMemoryConfig, *, mode: str) -> None:
        self.cfg = cfg
        self.mode = mode.strip().lower()
        self.records: list[SalesRecord] = []
        self.current_records: list[SalesRecord] = []
        self.contexts: dict[int, SalesContext] = {}
        self.seen_summary_instances: set[int] = set()
        self.seen_feedback_keys: set[tuple[int, str, str]] = set()
        self.catalog_by_id: dict[str, dict[str, Any]] = {}
        self.catalog_by_name: dict[str, dict[str, Any]] = {}
        self.locations_by_id: dict[str, dict[str, Any]] = {}
        self.locations_by_name: dict[str, dict[str, Any]] = {}
        self.write_count = 0
        self.prediction_count = 0
        self.last_update: dict[str, Any] = {}
        self.recent_contexts: list[dict[str, Any]] = []
        self.recent_predictions: list[dict[str, Any]] = []

    # ── CLBench command / prompt handling ────────────────────────────

    def begin_context(self, prompt: str, metadata: dict[str, Any]) -> SalesContext:
        instance_idx = _as_int(metadata.get("instance_idx"), 0)
        active_instance_idx = _as_int(metadata.get("active_instance_idx"), instance_idx)
        target_year = _as_int(metadata.get("target_year"), 0)
        raw_years = metadata.get("forecast_years") or []
        forecast_years = tuple(_as_int(y) for y in raw_years) or tuple(
            sorted({entry.year for entry in _parse_required_entries(prompt)})
        )
        required = tuple(_parse_required_entries(prompt))
        if not required and instance_idx in self.contexts:
            required = self.contexts[instance_idx].required
        if target_year == 0 and forecast_years:
            target_year = int(forecast_years[0])
        step = str(metadata.get("step", ""))
        feedback_rows = tuple(_parse_feedback_rows(prompt, instance_idx=instance_idx))
        context = SalesContext(
            instance_idx=instance_idx,
            active_instance_idx=active_instance_idx,
            target_year=target_year,
            forecast_years=tuple(int(y) for y in forecast_years),
            step=step,
            required=required,
            feedback_rows=feedback_rows,
        )
        if required:
            self.contexts[instance_idx] = context
        self._observe_feedback(context)
        self.recent_contexts.append(
            {
                "instance_idx": context.instance_idx,
                "target_year": context.target_year,
                "forecast_years": list(context.forecast_years),
                "required_count": len(context.required),
                "feedback_rows": len(context.feedback_rows),
            }
        )
        self.recent_contexts = self.recent_contexts[-16:]
        return context

    def needs_data_inspection(self, context: SalesContext) -> bool:
        mode = self.mode
        if mode == "sales_static_policy":
            return False
        return context.instance_idx not in self.seen_summary_instances

    def inspection_command(self) -> str:
        return _INSPECTION_COMMAND

    def submit_command(self) -> str:
        return "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

    def observe_command_output(self, text: str, context: SalesContext | None) -> None:
        if context is None:
            return
        payload = _extract_summary_payload(text)
        if payload is None:
            return
        records = self._records_from_payload(payload, context)
        self.current_records = records
        self.seen_summary_instances.add(context.instance_idx)
        mode = self.mode
        write_enabled = mode in {
            "sales_energy",
            "sales_vanilla_online",
            "sales_hard_cache",
            "sales_sliding_window",
        }
        if write_enabled:
            self._write_records(records)
        self.last_update = {
            "kind": "sales_summary",
            "mode": self.mode,
            "instance_idx": context.instance_idx,
            "records_seen": len(records),
            "records_total": len(self.records),
            "catalog_size": len(self.catalog_by_name),
            "locations": len(self.locations_by_name),
        }

    # ── Prediction ───────────────────────────────────────────────────

    def build_prediction_action(
        self,
        *,
        schema: type[Any],
        context: SalesContext,
    ) -> Any:
        readout = self._readout(context)
        entries = list(context.required)
        if not entries:
            entries = _entries_from_schema(schema)
        predictions = []
        for entry in entries:
            predictions.append(
                {
                    "locality": entry.locality,
                    "furniture_name": entry.furniture_name,
                    "year": int(entry.year),
                    "items_sold": round(max(0.0, self._predict_entry(entry, readout)), 3),
                }
            )
        self.prediction_count += 1
        snapshot = {
            "instance_idx": context.instance_idx,
            "mode": self.mode,
            "prediction_count": len(predictions),
            "read_policy": readout.read_policy,
            "record_count": readout.record_count,
            "growth_by_cluster": {
                k: round(v, 6) for k, v in sorted(readout.growth_by_cluster.items())
            },
        }
        self.recent_predictions.append(snapshot)
        self.recent_predictions = self.recent_predictions[-16:]
        self.last_update = {"kind": "sales_prediction", **snapshot}
        return schema(predictions=predictions)

    def artifacts(self) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        for rec in self.records:
            source_counts[rec.source] = source_counts.get(rec.source, 0) + 1
        return {
            "mode": self.mode,
            "record_count": len(self.records),
            "current_record_count": len(self.current_records),
            "catalog_size": len(self.catalog_by_name),
            "location_count": len(self.locations_by_name),
            "write_count": self.write_count,
            "prediction_count": self.prediction_count,
            "seen_summary_instances": sorted(self.seen_summary_instances),
            "source_counts": source_counts,
            "last_update": dict(self.last_update),
            "recent_contexts": list(self.recent_contexts[-8:]),
            "recent_predictions": list(self.recent_predictions[-8:]),
            "landscape_metrics": self._landscape_metrics(),
        }

    # ── Internal write/read paths ────────────────────────────────────

    def _observe_feedback(self, context: SalesContext) -> None:
        mode = self.mode
        if mode in {"sales_static_policy"}:
            return
        rows: list[SalesRecord] = []
        for row in context.feedback_rows:
            locality = str(row.get("locality") or "")
            furniture_name = str(row.get("furniture_name") or "")
            year = _as_int(row.get("year"), context.target_year - 1)
            key = (year, locality, furniture_name)
            if key in self.seen_feedback_keys:
                continue
            actual = _as_float(row.get("actual"), default=float("nan"))
            if not math.isfinite(actual):
                continue
            item = self.catalog_by_name.get(_norm(furniture_name), {})
            rec = SalesRecord(
                locality=locality,
                location_id=str((self.locations_by_name.get(_norm(locality), {}) or {}).get("location_id", "")),
                furniture_name=furniture_name,
                furniture_id=str(item.get("furniture_id", "")),
                furniture_type=str(item.get("furniture_type", "")),
                cluster=_semantic_cluster(str(item.get("furniture_type", "")), furniture_name),
                price=_as_float(item.get("furniture_price"), 500.0),
                year=year,
                items_sold=actual,
                source="feedback",
                instance_idx=context.instance_idx,
            )
            rows.append(rec)
            self.seen_feedback_keys.add(key)
        if not rows:
            return
        if mode in {
            "sales_energy",
            "sales_vanilla_online",
            "sales_hard_cache",
            "sales_sliding_window",
            "sales_no_write",
            "sales_current_instance_only",
        }:
            if mode in {"sales_no_write", "sales_current_instance_only"}:
                self.current_records.extend(rows)
            else:
                self._write_records(rows)

    def _write_records(self, records: list[SalesRecord]) -> None:
        if not records:
            return
        by_key: dict[tuple[str, str, int], SalesRecord] = {rec.key(): rec for rec in self.records}
        for rec in records:
            by_key[rec.key()] = rec
        merged = list(by_key.values())
        merged.sort(key=lambda r: (r.instance_idx, r.year, r.locality, r.furniture_name))
        if self.mode == "sales_sliding_window":
            latest = max((r.instance_idx for r in merged), default=0)
            floor = latest - max(1, self.cfg.sliding_window) + 1
            merged = [r for r in merged if r.instance_idx >= floor]
        if len(merged) > self.cfg.max_records:
            merged = merged[-self.cfg.max_records :]
        self.records = merged
        self.write_count += len(records)

    def _readout(self, context: SalesContext) -> SalesReadout:
        mode = self.mode
        if mode == "sales_static_policy":
            records: list[SalesRecord] = []
            policy = "static_catalog_prior"
        elif mode in {"sales_no_write", "sales_current_instance_only"}:
            records = list(self.current_records)
            policy = "current_public_data_only"
        elif mode == "sales_hard_cache":
            records = list(self.records)
            policy = "exact_entity_cache_then_cluster"
        elif mode == "sales_sliding_window":
            records = list(self.records)
            policy = f"sliding_window_{self.cfg.sliding_window}"
        elif mode == "sales_vanilla_online":
            records = list(self.records)
            policy = "full_horizon_online_statistics"
        else:
            records = list(self.records)
            policy = "energy_weighted_entity_cluster_readout"
        growth = self._growth_by_cluster(records)
        source_counts: dict[str, int] = {}
        for rec in records:
            source_counts[rec.source] = source_counts.get(rec.source, 0) + 1
        return SalesReadout(
            mode=mode,
            read_policy=policy,
            record_count=len(records),
            current_record_count=len(self.current_records),
            feedback_record_count=source_counts.get("feedback", 0),
            growth_by_cluster=growth,
            source_counts=source_counts,
        )

    def _predict_entry(self, entry: RequiredSalesEntry, readout: SalesReadout) -> float:
        item = self.catalog_by_name.get(_norm(entry.furniture_name), {})
        ftype = str(item.get("furniture_type", ""))
        price = _as_float(item.get("furniture_price"), 600.0)
        cluster = _semantic_cluster(ftype, entry.furniture_name)
        records = self._records_for_read(readout)

        if not records:
            return self._static_prior(price=price, year=entry.year)

        exact = [
            rec
            for rec in records
            if _norm(rec.locality) == _norm(entry.locality)
            and _norm(rec.furniture_name) == _norm(entry.furniture_name)
            and rec.year < entry.year
        ]
        if exact:
            base = max(exact, key=lambda r: (r.year, r.instance_idx))
            growth = self._entity_growth(exact, default=readout.growth_by_cluster.get(cluster, self.cfg.default_growth))
            return base.items_sold * math.exp(growth * max(0, entry.year - base.year))

        cluster_records = [
            rec
            for rec in records
            if rec.cluster == cluster
            and _norm(rec.locality) == _norm(entry.locality)
            and rec.year < entry.year
        ]
        if not cluster_records:
            cluster_records = [
                rec for rec in records if rec.cluster == cluster and rec.year < entry.year
            ]
        if not cluster_records:
            cluster_records = [rec for rec in records if rec.year < entry.year]
        if not cluster_records:
            return self._static_prior(price=price, year=entry.year)

        base_year = max(rec.year for rec in cluster_records)
        bucket_records = [rec for rec in cluster_records if rec.year == base_year]
        if not bucket_records:
            bucket_records = cluster_records
        base = self._price_adjusted_level(bucket_records, target_price=price)
        growth = readout.growth_by_cluster.get(cluster, self.cfg.default_growth)
        if readout.mode == "sales_hard_cache" and not exact:
            growth = self.cfg.default_growth
        return base * math.exp(growth * max(0, entry.year - base_year))

    def _records_for_read(self, readout: SalesReadout) -> list[SalesRecord]:
        if readout.mode in {"sales_no_write", "sales_current_instance_only"}:
            return list(self.current_records)
        if readout.mode == "sales_static_policy":
            return []
        return list(self.records)

    def _records_from_payload(self, payload: dict[str, Any], context: SalesContext) -> list[SalesRecord]:
        self._update_catalogs(payload)
        records: list[SalesRecord] = []
        for row in payload.get("sales_rows") or []:
            if not isinstance(row, dict):
                continue
            fid = str(_first(row, _FURNITURE_ID_KEYS) or "")
            loc_id = str(_first(row, _LOCATION_ID_KEYS) or "")
            year = _as_int(_first(row, _YEAR_KEYS), default=0)
            items = _as_float(_first(row, _ITEMS_KEYS), default=float("nan"))
            if not fid or not loc_id or year <= 0 or not math.isfinite(items):
                continue
            item = self.catalog_by_id.get(fid, {})
            loc = self.locations_by_id.get(loc_id, {})
            furniture_name = str(item.get("furniture_name") or fid)
            ftype = str(item.get("furniture_type") or "")
            price = _as_float(item.get("furniture_price"), 600.0)
            locality = str(loc.get("locality") or loc_id)
            records.append(
                SalesRecord(
                    locality=locality,
                    location_id=loc_id,
                    furniture_name=furniture_name,
                    furniture_id=fid,
                    furniture_type=ftype,
                    cluster=_semantic_cluster(ftype, furniture_name),
                    price=price,
                    year=year,
                    items_sold=items,
                    source=str(row.get("file") or "sales_csv"),
                    instance_idx=context.instance_idx,
                )
            )
        return records

    def _update_catalogs(self, payload: dict[str, Any]) -> None:
        for item in payload.get("catalog") or []:
            if not isinstance(item, dict):
                continue
            fid = str(_first(item, _FURNITURE_ID_KEYS) or item.get("id") or "")
            name = str(
                item.get("furniture_name")
                or item.get("product_name")
                or item.get("description")
                or fid
            )
            ftype = str(
                item.get("furniture_type")
                or item.get("category")
                or item.get("item_category")
                or ""
            )
            price = _as_float(
                item.get("furniture_price")
                or item.get("price")
                or item.get("retail_price"),
                600.0,
            )
            normalized = {
                "furniture_id": fid,
                "furniture_name": name,
                "furniture_type": ftype,
                "furniture_price": price,
                "cluster": _semantic_cluster(ftype, name),
            }
            if fid:
                self.catalog_by_id[fid] = normalized
            if name:
                self.catalog_by_name[_norm(name)] = normalized
        for loc in payload.get("locations") or []:
            if not isinstance(loc, dict):
                continue
            loc_id = str(_first(loc, _LOCATION_ID_KEYS) or "")
            locality = str(
                loc.get("locality")
                or loc.get("city")
                or loc.get("branch_city")
                or loc_id
            )
            normalized = {
                "location_id": loc_id,
                "locality": locality,
                "state": loc.get("state") or loc.get("branch_state") or "",
            }
            if loc_id:
                self.locations_by_id[loc_id] = normalized
            if locality:
                self.locations_by_name[_norm(locality)] = normalized

    def _growth_by_cluster(self, records: list[SalesRecord]) -> dict[str, float]:
        by_entity: dict[tuple[str, str], list[SalesRecord]] = {}
        for rec in records:
            by_entity.setdefault(rec.entity(), []).append(rec)
        by_cluster: dict[str, list[float]] = {}
        for seq in by_entity.values():
            seq = sorted(seq, key=lambda r: r.year)
            for left, right in zip(seq, seq[1:]):
                dt = right.year - left.year
                if dt <= 0 or left.items_sold <= 0 or right.items_sold <= 0:
                    continue
                g = (math.log(right.items_sold) - math.log(left.items_sold)) / dt
                g = min(self.cfg.max_growth, max(self.cfg.min_growth, g))
                by_cluster.setdefault(right.cluster, []).append(g)
        growth: dict[str, float] = {}
        for cluster, vals in by_cluster.items():
            if not vals:
                continue
            vals_sorted = sorted(vals)
            growth[cluster] = vals_sorted[len(vals_sorted) // 2]
        for rec in records:
            growth.setdefault(rec.cluster, self.cfg.default_growth)
        return growth

    def _entity_growth(self, records: list[SalesRecord], *, default: float) -> float:
        seq = sorted(records, key=lambda r: r.year)
        vals = []
        for left, right in zip(seq, seq[1:]):
            dt = right.year - left.year
            if dt <= 0 or left.items_sold <= 0 or right.items_sold <= 0:
                continue
            vals.append((math.log(right.items_sold) - math.log(left.items_sold)) / dt)
        if not vals:
            return default
        vals = [min(self.cfg.max_growth, max(self.cfg.min_growth, v)) for v in vals]
        vals.sort()
        return vals[len(vals) // 2]

    def _price_adjusted_level(self, records: list[SalesRecord], *, target_price: float) -> float:
        if not records:
            return self._static_prior(price=target_price, year=0)
        implied = []
        for rec in records:
            demand_ex_floor = max(1.0, rec.items_sold - self.cfg.demand_floor)
            ceiling = demand_ex_floor * (1.0 + math.exp(rec.price / self.cfg.price_scale))
            implied.append(ceiling)
        implied.sort()
        ceiling_est = implied[len(implied) // 2]
        return self.cfg.demand_floor + ceiling_est / (1.0 + math.exp(target_price / self.cfg.price_scale))

    def _static_prior(self, *, price: float, year: int) -> float:
        base = self.cfg.demand_floor + 700.0 / (1.0 + math.exp(price / self.cfg.price_scale))
        if year:
            base *= math.exp(self.cfg.default_growth * max(0, year - 2026))
        return base

    def _landscape_metrics(self) -> dict[str, Any]:
        by_cluster: dict[str, int] = {}
        by_locality: dict[str, int] = {}
        years = []
        for rec in self.records:
            by_cluster[rec.cluster] = by_cluster.get(rec.cluster, 0) + 1
            by_locality[rec.locality] = by_locality.get(rec.locality, 0) + 1
            years.append(rec.year)
        return {
            "cluster_counts": by_cluster,
            "locality_counts": by_locality,
            "min_year": min(years) if years else None,
            "max_year": max(years) if years else None,
        }


def _parse_required_entries(prompt: str) -> list[RequiredSalesEntry]:
    entries: list[RequiredSalesEntry] = []
    pat = re.compile(r"^\s*-\s*(?P<loc>.+?)\s*/\s*(?P<name>.+?)\s*/\s*(?P<year>\d{4})\s*$")
    for line in prompt.splitlines():
        match = pat.match(line)
        if not match:
            continue
        entries.append(
            RequiredSalesEntry(
                locality=match.group("loc").strip(),
                furniture_name=match.group("name").strip(),
                year=int(match.group("year")),
            )
        )
    return entries


def _parse_feedback_rows(prompt: str, *, instance_idx: int) -> list[dict[str, Any]]:
    if "Previous round feedback" not in prompt:
        return []
    blocks = re.findall(r"```json\s*(.*?)\s*```", prompt, flags=re.DOTALL | re.IGNORECASE)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        year = _as_int(payload.get("feedback_year"), 0)
        for entry in payload.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "year": year,
                    "locality": entry.get("locality"),
                    "furniture_name": entry.get("furniture_name"),
                    "actual": entry.get("actual"),
                    "source": "prompt_feedback",
                    "instance_idx": instance_idx,
                }
            )
    return rows


def _extract_summary_payload(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if _SUMMARY_MARKER not in line:
            continue
        raw = line.split(_SUMMARY_MARKER, 1)[1].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _entries_from_schema(schema: type[Any]) -> list[RequiredSalesEntry]:
    """Best-effort fallback for dynamic Literal-constrained schemas."""
    fields = getattr(schema, "model_fields", {})
    pred_field = fields.get("predictions") if isinstance(fields, dict) else None
    annotation = getattr(pred_field, "annotation", None)
    args = getattr(annotation, "__args__", ()) if annotation is not None else ()
    if not args:
        return []
    entry_cls = args[0]
    entry_fields = getattr(entry_cls, "model_fields", {})
    def literals(name: str) -> list[Any]:
        ann = getattr(entry_fields.get(name), "annotation", None)
        return list(getattr(ann, "__args__", ()) or [])
    localities = [str(v) for v in literals("locality")]
    names = [str(v) for v in literals("furniture_name")]
    years = [_as_int(v) for v in literals("year")]
    # The dynamic schema does not encode the cross product mask, so this fallback
    # is intentionally conservative.  Normal operation parses exact prompt lines.
    if not (localities and names and years):
        return []
    return [
        RequiredSalesEntry(loc, name, year)
        for year in years
        for loc in localities
        for name in names
    ]


_INSPECTION_COMMAND = r"""python3 - <<'PY'
import csv, json
from pathlib import Path

base = Path('/app/data')
payload = {
    'files': {},
    'catalog': [],
    'types': {},
    'locations': [],
    'sales_rows': [],
}

def load_json(name, default):
    path = base / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

payload['catalog'] = load_json('furniture.json', [])
payload['types'] = load_json('furniture_types.json', {})
payload['locations'] = load_json('locations.json', [])

for path in sorted(base.glob('*.csv')):
    try:
        with path.open(newline='') as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            payload['files'][path.name] = {
                'columns': list(reader.fieldnames or []),
                'rows': len(rows),
            }
            for row in rows:
                item = {'file': path.name}
                item.update(row)
                payload['sales_rows'].append(item)
    except Exception as exc:
        payload['files'][path.name] = {'error': str(exc)}

print('NINED_SALES_SUMMARY_JSON=' + json.dumps(payload, separators=(',', ':')))
PY"""
