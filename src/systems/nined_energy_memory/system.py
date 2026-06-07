"""CLBench adapter for the 9D bounded energy-memory runtime."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import re

from pydantic import BaseModel

from ...interface import ContinualLearningSystem, Observation, Query, Response
from ...registry import register_system
from .cohort_planner import choose_cohort_tool_action, parse_cohort_context
from .cohort_runtime import CohortMemoryConfig, CohortMemoryRuntime
from .cohort_submission import build_cohort_submission
from .db_planner import choose_database_action, parse_database_context
from .db_runtime import DatabaseMemoryConfig, DatabaseMemoryRuntime
from .poker_runtime import (
    PokerMemoryConfig,
    PokerMemoryRuntime,
    choose_poker_action,
    parse_hand_delta,
    parse_poker_context,
)
from .runtime import (
    BandInfo,
    SpectrumEnergyConfig,
    SpectrumEnergyMemoryRuntime,
    SpectrumPeak,
)
from .sales_runtime import SalesContext, SalesMemoryConfig, SalesMemoryRuntime


PEAK_RE = re.compile(
    r"-\s*peak_id:\s*(?P<peak_id>[^|]+)\|\s*"
    r"freq:\s*(?P<freq>-?\d+(?:\.\d+)?)\s*MHz\s*\|\s*"
    r"power:\s*(?P<power>-?\d+(?:\.\d+)?)\s*dBm\s*\|\s*"
    r"width:\s*(?P<width>-?\d+(?:\.\d+)?)\s*MHz",
    re.IGNORECASE,
)
BAND_RE = re.compile(
    r"Band:\s*(?P<start>-?\d+(?:\.\d+)?)\s*-\s*(?P<end>-?\d+(?:\.\d+)?)\s*MHz",
    re.IGNORECASE,
)
NOISE_RE = re.compile(
    r"estimated_noise_floor_dbm:\s*(?P<noise>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@register_system("nined_energy_memory")
class NineDEnergyMemorySystem(ContinualLearningSystem):
    """Registered CLBench system that evaluates the energy-memory runtime."""

    supports_baseline = True
    parallel_safe = True

    def __init__(
        self,
        mode: str = "bsm_energy",
        name: str = "nined_energy_memory",
        capacity: int = 32,
        write_lr: float = 1.15,
        report_logit_threshold: float = 0.65,
        min_report_hits: int = 2,
        merge_radius_mhz: float = 4.0,
        max_report: int = 24,
        sliding_window: int = 8,
        poker_capacity: int = 8,
        poker_sliding_window: int = 20,
        poker_use_opponent_name: bool = True,
        db_capacity: int = 64,
        db_fact_capacity: int = 512,
        db_manifold_dim: int = 64,
        db_spectral_k: int = 16,
        db_chebyshev_order: int = 4,
        db_write_lr: float = 0.35,
        db_answer_confidence_threshold: float = 0.72,
        db_max_explore_queries: int = 4,
        cohort_capacity: int = 96,
        cohort_fact_capacity: int = 2048,
        cohort_manifold_dim: int = 96,
        cohort_read_temperature: float = 0.70,
        cohort_sliding_window: int = 6,
        cohort_write_lr: float = 0.30,
        cohort_max_tool_steps: int = 20,
        cohort_profile_count: int = 4,
        sales_sliding_window: int = 4,
        sales_max_records: int = 4096,
        sales_default_growth: float = 0.14842,
    ) -> None:
        self.mode = mode
        self._name = name
        self.cfg = SpectrumEnergyConfig(
            capacity=capacity,
            write_lr=write_lr,
            report_logit_threshold=report_logit_threshold,
            min_report_hits=min_report_hits,
            merge_radius_mhz=merge_radius_mhz,
            max_report=max_report,
            sliding_window=sliding_window,
        )
        self.poker_cfg = PokerMemoryConfig(
            capacity=poker_capacity,
            sliding_window=poker_sliding_window,
        )
        self.db_cfg = DatabaseMemoryConfig(
            capacity=db_capacity,
            fact_capacity=db_fact_capacity,
            manifold_dim=db_manifold_dim,
            spectral_k=db_spectral_k,
            chebyshev_order=db_chebyshev_order,
            write_lr=db_write_lr,
            answer_confidence_threshold=db_answer_confidence_threshold,
            max_queries_per_question=max(1, db_max_explore_queries),
        )
        self.cohort_cfg = CohortMemoryConfig(
            capacity=cohort_capacity,
            fact_capacity=cohort_fact_capacity,
            manifold_dim=cohort_manifold_dim,
            read_temperature=cohort_read_temperature,
            sliding_window=cohort_sliding_window,
            write_lr=cohort_write_lr,
            profile_count=cohort_profile_count,
        )
        self.sales_cfg = SalesMemoryConfig(
            sliding_window=sales_sliding_window,
            max_records=sales_max_records,
            default_growth=sales_default_growth,
        )
        self.cohort_max_tool_steps = max(1, cohort_max_tool_steps)
        self.poker_use_opponent_name = poker_use_opponent_name
        self._runtime = self._make_runtime()
        self._poker_runtime = self._make_poker_runtime()
        self._db_runtime = self._make_db_runtime()
        self._cohort_runtime = self._make_cohort_runtime()
        self._sales_runtime = self._make_sales_runtime()
        self._interaction_count = 0
        self._last_response_metadata: dict[str, Any] = {}
        self._observations: list[dict[str, Any]] = []
        self._last_poker_identity_key: str | None = None
        self._last_poker_decision: dict[str, Any] = {}
        self._last_db_action: dict[str, Any] = {}
        self._last_db_query_context: dict[str, Any] = {}
        self._last_cohort_action: dict[str, Any] = {}
        self._last_cohort_query_context: dict[str, Any] = {}
        self._last_sales_context: SalesContext | None = None

    @property
    def name(self) -> str:
        return self._name

    def reset(self) -> None:
        self._runtime = self._make_runtime()
        self._poker_runtime = self._make_poker_runtime()
        self._db_runtime = self._make_db_runtime()
        self._cohort_runtime = self._make_cohort_runtime()
        self._sales_runtime = self._make_sales_runtime()
        self._interaction_count = 0
        self._last_response_metadata = {}
        self._observations = []
        self._last_poker_identity_key = None
        self._last_poker_decision = {}
        self._last_db_action = {}
        self._last_db_query_context = {}
        self._last_cohort_action = {}
        self._last_cohort_query_context = {}
        self._last_sales_context = None

    def respond(self, query: Query) -> Response:
        self._interaction_count += 1

        if self._is_bsm_query(query):
            return self._respond_bsm(query)

        if self._is_poker_query(query):
            return self._respond_poker(query)

        if self._is_database_query(query):
            return self._respond_database(query)

        if self._is_cohort_tool_query(query):
            return self._respond_cohort_tool(query)

        if self._is_cohort_submission_query(query):
            return self._respond_cohort_submission(query)

        if self._is_sales_command_query(query):
            return self._respond_sales_command(query)

        if self._is_sales_prediction_query(query):
            return self._respond_sales_prediction(query)

        raise NotImplementedError(
            "nined_energy_memory currently implements blind_spectrum_monitoring "
            "exploitable_poker, database_exploration, cohort_studies, and sales_prediction "
            "attribution modes. LLM-backed modes are planned but not yet implemented."
        )

    def observe(
        self, observation: Observation, next_query: Query | None = None
    ) -> None:
        self._observations.append(
            {
                "content": observation.content,
                "instance_complete": observation.instance_complete,
                "metadata": observation.metadata or {},
                "next_instance_index": None
                if next_query is None
                else next_query.instance_index,
            }
        )
        # Keep artifacts bounded.
        if len(self._observations) > 16:
            self._observations = self._observations[-16:]

        if self.mode.strip().lower().startswith("poker_"):
            self._observe_poker(observation)
        if self.mode.strip().lower().startswith("db_"):
            self._observe_database(observation)
        if self.mode.strip().lower().startswith("cohort_"):
            self._observe_cohort(observation)
        if self.mode.strip().lower().startswith("sales_"):
            self._observe_sales(observation)

    def get_run_artifacts(self) -> dict[str, Any]:
        return {
            "artifact_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "last_response_metadata": dict(self._last_response_metadata),
            "recent_observations": list(self._observations),
            "runtime": self._runtime.artifacts(),
            "poker_runtime": self._poker_runtime.artifacts(),
            "db_runtime": self._db_runtime.artifacts(),
            "cohort_runtime": self._cohort_runtime.artifacts(),
            "sales_runtime": self._sales_runtime.artifacts(),
        }

    def _make_runtime(self) -> SpectrumEnergyMemoryRuntime:
        mode = self.mode.strip().lower()
        write_enabled = mode not in {
            "bsm_no_write",
            "bsm_current_scan_only",
            "bsm_static_pipeline",
            "bsm_vanilla_online",
        }
        propagation_enabled = mode != "bsm_no_propagation"
        sliding_window_enabled = mode == "bsm_sliding_window"
        return SpectrumEnergyMemoryRuntime(
            self.cfg,
            write_enabled=write_enabled,
            propagation_enabled=propagation_enabled,
            sliding_window_enabled=sliding_window_enabled,
        )

    def _make_poker_runtime(self) -> PokerMemoryRuntime:
        return PokerMemoryRuntime(
            self.poker_cfg,
            mode=self.mode.strip().lower(),
            use_opponent_name=self.poker_use_opponent_name,
        )

    def _make_db_runtime(self) -> DatabaseMemoryRuntime:
        return DatabaseMemoryRuntime(
            self.db_cfg,
            mode=self.mode.strip().lower(),
        )

    def _make_cohort_runtime(self) -> CohortMemoryRuntime:
        return CohortMemoryRuntime(
            self.cohort_cfg,
            mode=self.mode.strip().lower(),
        )

    def _make_sales_runtime(self) -> SalesMemoryRuntime:
        return SalesMemoryRuntime(
            self.sales_cfg,
            mode=self.mode.strip().lower(),
        )

    def _is_bsm_query(self, query: Query) -> bool:
        schema_name = getattr(query.response_schema, "__name__", "")
        return schema_name == "ScanReport" or "Detected peaks:" in query.prompt

    def _is_poker_query(self, query: Query) -> bool:
        schema_name = getattr(query.response_schema, "__name__", "")
        return schema_name == "PokerAction" or "Heads-up Texas Hold'em" in query.prompt

    def _is_database_query(self, query: Query) -> bool:
        schema_name = getattr(query.response_schema, "__name__", "")
        return schema_name == "DatabaseAction" or "unknown database" in query.prompt

    def _is_cohort_tool_query(self, query: Query) -> bool:
        schema_name = getattr(query.response_schema, "__name__", "")
        metadata = query.metadata or {}
        return schema_name == "ToolCallResponse" and (
            "cohort" in query.prompt.lower()
            or "clinical stud" in query.prompt.lower()
            or metadata.get("study_name") is not None
        )

    def _is_cohort_submission_query(self, query: Query) -> bool:
        schema_name = getattr(query.response_schema, "__name__", "")
        fields = getattr(query.response_schema, "model_fields", {})
        return schema_name == "CohortSubmission" or any(
            str(name).endswith("__s12") for name in fields
        )

    def _is_sales_command_query(self, query: Query) -> bool:
        schema_name = getattr(query.response_schema, "__name__", "")
        metadata = query.metadata or {}
        return schema_name == "BashCommandResponse" and (
            "sales_prediction" in str(query.instance_id or "")
            or metadata.get("target_year") is not None
            or "furniture retailer" in query.prompt.lower()
            or "sales_historical.csv" in query.prompt
        )

    def _is_sales_prediction_query(self, query: Query) -> bool:
        schema_name = getattr(query.response_schema, "__name__", "")
        metadata = query.metadata or {}
        return schema_name == "PredictionResponse" and (
            metadata.get("step") == "prediction_extraction"
            or "Required entries" in query.prompt
            or "items_sold" in query.prompt
        )

    def _respond_bsm(self, query: Query) -> Response:
        peaks = self._parse_peaks(query.prompt)
        band = self._parse_band(query.prompt)
        scan_index = self._scan_index(query)
        mode = self.mode.strip().lower()

        if mode in {"bsm_current_scan_only", "bsm_static_pipeline"}:
            transmitter_dicts = self._runtime.current_scan_reports(peaks, band=band)
        else:
            self._runtime.update_from_scan(peaks, band=band, scan_index=scan_index)
            if mode == "bsm_no_write":
                transmitter_dicts = self._runtime.current_scan_reports(peaks, band=band)
            elif mode == "bsm_vanilla_online":
                transmitter_dicts = self._runtime.vanilla_online_reports(band=band)
            else:
                transmitter_dicts = self._runtime.reports(band=band)

        action = self._make_action(query.response_schema, transmitter_dicts)
        metadata = {
            "system_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "scan_index": scan_index,
            "num_peaks": len(peaks),
            "num_reported": len(transmitter_dicts),
            "runtime_last_update": dict(self._runtime.last_update),
        }
        self._last_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def _respond_poker(self, query: Query) -> Response:
        mode = self.mode.strip().lower()
        context = parse_poker_context(
            prompt=query.prompt,
            metadata=query.metadata or {},
            use_opponent_name=self.poker_use_opponent_name,
        )

        write_enabled = mode in {
            "poker_energy",
            "poker_hard_cache",
            "poker_vanilla_online",
            "poker_sliding_window",
        }
        if write_enabled:
            self._poker_runtime.update_from_prompt(context)

        stats = self._poker_runtime.stats_for(context)
        action_name, amount, reason = choose_poker_action(
            context=context,
            stats=stats,
        )
        action = query.response_schema(
            thinking=reason,
            action=action_name,
            amount=amount if action_name == "RAISE" else None,
        )
        self._last_poker_identity_key = context.identity_key
        metadata = {
            "system_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "task": "exploitable_poker",
            "identity_mode": context.identity_mode,
            "identity_key": context.identity_key,
            "opponent_name_visible": self.poker_use_opponent_name,
            "hand_id": context.hand_id,
            "hand_num": context.hand_num,
            "phase": context.phase,
            "board_texture": context.board_texture,
            "strength": round(context.strength, 4),
            "legal_actions": list(context.legal_actions),
            "action": action_name,
            "amount": amount,
            "pot": context.pot,
            "chips_to_call": context.chips_to_call,
            "player_chips": context.player_chips,
            "stats": {key: round(value, 6) for key, value in stats.items()},
            "runtime_last_update": dict(self._poker_runtime.last_update),
        }
        self._last_poker_decision = {
            "identity_key": context.identity_key,
            "action": action_name,
            "amount": amount,
            "phase": context.phase,
            "board_texture": context.board_texture,
            "strength": context.strength,
            "pot": context.pot,
            "chips_to_call": context.chips_to_call,
        }
        self._last_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def _respond_database(self, query: Query) -> Response:
        context = parse_database_context(query.prompt, query.metadata or {})
        readout = self._db_runtime.begin_question(context)
        plan = choose_database_action(context=context, readout=readout)
        action = query.response_schema(action=plan.action, content=plan.content)
        metadata = {
            "system_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "task": "database_exploration",
            "question_id": context.question_id,
            "question_num": context.question_num,
            "stage": context.stage,
            "queries_used": context.queries_used,
            "schema_known": readout.schema_known,
            "answer_confidence": round(plan.answer_confidence, 5),
            "db_action": plan.action,
            "db_sql_kind": plan.sql_kind,
            "db_reason": plan.reason,
            "top_fact_ids": [fact.get("fact_id") for fact in readout.top_facts[:8]],
            "runtime_last_update": dict(self._db_runtime.last_update),
        }
        self._last_db_action = plan.to_artifact()
        self._last_db_query_context = {
            "question_id": context.question_id,
            "question_num": context.question_num,
            "stage": context.stage,
        }
        self._last_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def _respond_cohort_tool(self, query: Query) -> Response:
        context = parse_cohort_context(query.prompt, query.metadata or {})
        readout = self._cohort_runtime.begin_turn(context, target_kind="tool")
        plan = choose_cohort_tool_action(
            context=context,
            readout=readout,
            max_tool_steps=self.cohort_max_tool_steps,
        )
        action = query.response_schema(thought=plan.reason, tool_call=plan.tool_call)
        metadata = {
            "system_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "task": "cohort_studies",
            "instance_idx": context.instance_idx,
            "study_name": context.study_name,
            "variant_id": context.variant_id,
            "step": context.step,
            "cohort_tool": plan.tool,
            "cohort_tool_signature": plan.signature,
            "cohort_reason": plan.reason,
            "read_policy": readout.read_policy,
            "top_fact_ids": [fact.get("fact_id") for fact in readout.top_facts[:8]],
            "top_basin_ids": [basin.get("basin_id") for basin in readout.top_basins[:8]],
            "read_entropy": readout.read_entropy,
            "free_energy": readout.free_energy,
            "energy_margin": readout.energy_margin,
            "visible_columns": list(readout.visible_columns),
            "runtime_last_update": dict(self._cohort_runtime.last_update),
        }
        self._last_cohort_action = plan.to_artifact()
        self._last_cohort_query_context = asdict(context)
        self._last_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def _respond_cohort_submission(self, query: Query) -> Response:
        context = parse_cohort_context(query.prompt, query.metadata or {})
        readout = self._cohort_runtime.begin_turn(
            context,
            target_kind="submission",
            target_id="all_cohorts",
        )
        action = build_cohort_submission(
            schema=query.response_schema,
            readout=readout,
            runtime=self._cohort_runtime,
        )
        metadata = {
            "system_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "task": "cohort_studies",
            "instance_idx": context.instance_idx,
            "study_name": context.study_name,
            "variant_id": context.variant_id,
            "phase": "cohort_submission",
            "read_policy": readout.read_policy,
            "top_fact_ids": [fact.get("fact_id") for fact in readout.top_facts[:8]],
            "top_basin_ids": [basin.get("basin_id") for basin in readout.top_basins[:8]],
            "read_entropy": readout.read_entropy,
            "free_energy": readout.free_energy,
            "energy_margin": readout.energy_margin,
            "runtime_last_update": dict(self._cohort_runtime.last_update),
        }
        self._last_cohort_action = {}
        self._last_cohort_query_context = {}
        self._last_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def _respond_sales_command(self, query: Query) -> Response:
        context = self._sales_runtime.begin_context(query.prompt, query.metadata or {})
        self._last_sales_context = context
        if self._sales_runtime.needs_data_inspection(context):
            command = self._sales_runtime.inspection_command()
            reason = "Inspect public /app/data files and emit compact sales summary."
            phase = "sales_data_inspection"
        else:
            command = self._sales_runtime.submit_command()
            reason = "Public data summary already observed; submit structured predictions."
            phase = "sales_submit"
        action = query.response_schema(thought=reason, command=command)
        metadata = {
            "system_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "task": "sales_prediction",
            "instance_idx": context.instance_idx,
            "target_year": context.target_year,
            "forecast_years": list(context.forecast_years),
            "required_count": len(context.required),
            "phase": phase,
            "runtime_last_update": dict(self._sales_runtime.last_update),
        }
        self._last_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def _respond_sales_prediction(self, query: Query) -> Response:
        context = self._sales_runtime.begin_context(query.prompt, query.metadata or {})
        self._last_sales_context = context
        action = self._sales_runtime.build_prediction_action(
            schema=query.response_schema,
            context=context,
        )
        metadata = {
            "system_type": "nined_energy_memory",
            "mode": self.mode,
            "interaction_count": self._interaction_count,
            "task": "sales_prediction",
            "instance_idx": context.instance_idx,
            "target_year": context.target_year,
            "forecast_years": list(context.forecast_years),
            "required_count": len(context.required),
            "phase": "sales_prediction",
            "runtime_last_update": dict(self._sales_runtime.last_update),
        }
        self._last_response_metadata = metadata
        return Response(action=action, metadata=metadata)

    def _observe_poker(self, observation: Observation) -> None:
        mode = self.mode.strip().lower()
        if mode not in {
            "poker_energy",
            "poker_hard_cache",
            "poker_vanilla_online",
            "poker_sliding_window",
        }:
            return
        delta = parse_hand_delta(observation.content)
        if delta is None:
            return
        decision = self._last_poker_decision
        self._poker_runtime.update_from_observation(
            identity_key=self._last_poker_identity_key,
            delta=delta,
            decision=str(decision.get("action")) if decision else None,
            phase=str(decision.get("phase")) if decision else None,
        )

    def _observe_database(self, observation: Observation) -> None:
        action = str(self._last_db_action.get("action", "")).upper()
        content = str(self._last_db_action.get("content", ""))
        if action == "QUERY":
            self._db_runtime.observe_query_result(
                sql=content,
                result_text=observation.content,
            )
        if observation.instance_complete:
            self._db_runtime.observe_answer_feedback(observation.content)

    def _observe_cohort(self, observation: Observation) -> None:
        action = self._last_cohort_action
        if not action:
            return
        tool = str(action.get("tool") or "")
        tool_call = action.get("tool_call") or {}
        if not isinstance(tool_call, dict):
            tool_call = {}
        ctx_payload = self._last_cohort_query_context
        if not ctx_payload:
            return
        context = parse_cohort_context("", ctx_payload)
        self._cohort_runtime.observe_tool_result(
            context=context,
            tool_name=tool,
            tool_payload=tool_call,
            result_text=observation.content,
        )

    def _observe_sales(self, observation: Observation) -> None:
        self._sales_runtime.observe_command_output(
            observation.content,
            self._last_sales_context,
        )

    def _make_action(
        self, schema: type[BaseModel], transmitter_dicts: list[dict[str, Any]]
    ) -> BaseModel:
        return schema(transmitters=transmitter_dicts)

    def _parse_peaks(self, prompt: str) -> list[SpectrumPeak]:
        peaks: list[SpectrumPeak] = []
        for match in PEAK_RE.finditer(prompt):
            peaks.append(
                SpectrumPeak(
                    peak_id=match.group("peak_id").strip(),
                    center_freq=float(match.group("freq")),
                    power_dbm=float(match.group("power")),
                    bandwidth=float(match.group("width")),
                )
            )
        return peaks

    def _parse_band(self, prompt: str) -> BandInfo:
        start = 0.0
        end = 180.0
        noise = -50.0
        band_match = BAND_RE.search(prompt)
        if band_match is not None:
            start = float(band_match.group("start"))
            end = float(band_match.group("end"))
        noise_match = NOISE_RE.search(prompt)
        if noise_match is not None:
            noise = float(noise_match.group("noise"))
        return BandInfo(start_mhz=start, end_mhz=end, noise_floor_dbm=noise)

    def _scan_index(self, query: Query) -> int:
        metadata = query.metadata or {}
        raw = metadata.get("active_instance_index", query.instance_index)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        return self._interaction_count - 1
