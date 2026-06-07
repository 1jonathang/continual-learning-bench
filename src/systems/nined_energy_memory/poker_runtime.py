"""Bounded online Poker runtime for the CLBench adapter.

This module is deliberately small and transparent.  It is not a poker engine;
it is an attribution scaffold for testing whether bounded online accumulation
helps on ``exploitable_poker`` beyond a static policy and beyond a plain
vanilla online estimator.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import math
import re
from typing import Any

try:  # pragma: no cover - optional poker extra
    from treys import Card as TreysCard
    from treys import Deck as TreysDeck
    from treys import Evaluator as TreysEvaluator
except ModuleNotFoundError:  # pragma: no cover - system validation without poker extra
    TreysCard = None  # type: ignore[assignment]
    TreysDeck = None  # type: ignore[assignment]
    TreysEvaluator = None  # type: ignore[assignment]


ACTION_NAMES = ("FOLD", "CALL", "CHECK", "RAISE")
STREET_ORDER = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}
TREYS_EVALUATOR = TreysEvaluator() if TreysEvaluator is not None else None
TREYS_FULL_DECK = tuple(TreysDeck.GetFullDeck()) if TreysDeck is not None else ()
RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

HAND_RE = re.compile(r"Your hand:\s*(?P<cards>.+)")
BOARD_RE = re.compile(r"Board:\s*(?P<cards>.+)")
OPP_ACTION_RE = re.compile(
    r"Opponent's actions?:\s*(?P<actions>[A-Z\s>\-]+)", re.IGNORECASE
)
NET_RE = re.compile(r"Net chip change this hand:\s*(?P<delta>[+-]?\d+)")
TOTAL_RE = re.compile(r"Total profit:\s*(?P<total>[+-]?\d+)")


@dataclass
class PokerMemoryConfig:
    """Hyperparameters for the Poker attribution runtime."""

    capacity: int = 8
    sliding_window: int = 20
    confidence_lr: float = 0.25
    confidence_decay: float = 0.995
    reward_lr: float = 0.20


@dataclass
class PokerContext:
    """Parsed state required to emit a legal Poker action."""

    hand_id: str
    hand_num: int
    phase: str
    opponent_name: str
    legal_actions: list[str]
    chips_to_call: int
    min_raise: int | None
    max_raise: int | None
    pot: int
    player_chips: int
    board_texture: str
    strength: float
    current_opponent_actions: list[str]
    identity_key: str
    identity_mode: str


@dataclass
class PokerOpponentModel:
    """One bounded opponent basin."""

    key: str
    created_at: int
    last_seen: int
    actions: Counter[str] = field(default_factory=Counter)
    street_actions: dict[str, Counter[str]] = field(default_factory=dict)
    situation_actions: dict[str, Counter[str]] = field(default_factory=dict)
    hands: int = 0
    total_profit: int = 0
    decision_counts: Counter[str] = field(default_factory=Counter)
    decision_reward_sum: Counter[str] = field(default_factory=Counter)
    street_decision_counts: dict[str, Counter[str]] = field(default_factory=dict)
    street_decision_reward_sum: dict[str, Counter[str]] = field(default_factory=dict)
    confidence_logit: float = 0.0

    def update_actions(
        self,
        actions: list[str],
        *,
        phase: str,
        situation_key: str,
        step: int,
    ) -> None:
        self.last_seen = step
        for action in actions:
            if action in ACTION_NAMES:
                self.actions[action] += 1
                self.street_actions.setdefault(phase, Counter())[action] += 1
                self.situation_actions.setdefault(situation_key, Counter())[action] += 1

    def update_profit(self, delta: int, *, step: int) -> None:
        self.last_seen = step
        self.hands += 1
        self.total_profit += int(delta)

    def update_decision_reward(
        self,
        *,
        decision: str | None,
        phase: str | None,
        delta: int,
        step: int,
    ) -> None:
        if not decision:
            return
        action = decision.upper()
        if action not in ACTION_NAMES:
            return
        self.last_seen = step
        self.decision_counts[action] += 1
        self.decision_reward_sum[action] += int(delta)
        street = str(phase or "UNKNOWN").upper()
        self.street_decision_counts.setdefault(street, Counter())[action] += 1
        self.street_decision_reward_sum.setdefault(street, Counter())[action] += int(
            delta
        )

    def action_total(self) -> int:
        return sum(self.actions.values())

    def rate(self, action: str) -> float:
        # Laplace smoothing avoids early hard decisions.
        return (self.actions[action] + 1.0) / (self.action_total() + len(ACTION_NAMES))


class PokerMemoryRuntime:
    """Bounded online opponent model used by Poker modes."""

    def __init__(
        self,
        cfg: PokerMemoryConfig | None = None,
        *,
        mode: str,
        use_opponent_name: bool = True,
    ) -> None:
        self.cfg = cfg or PokerMemoryConfig()
        self.mode = mode
        self.use_opponent_name = use_opponent_name
        self.step_index = 0
        self.models: list[PokerOpponentModel] = []
        self.recent_events: deque[tuple[str, str, str]] = deque()
        self.last_update: dict[str, Any] = {}

    def reset(self) -> None:
        self.step_index = 0
        self.models = []
        self.recent_events = deque()
        self.last_update = {}

    def update_from_prompt(self, context: PokerContext) -> None:
        """Absorb visible opponent actions since the last prompt."""

        if not context.current_opponent_actions:
            return
        self.step_index += 1
        model = self._model_for_key(context.identity_key)
        model.update_actions(
            context.current_opponent_actions,
            phase=context.phase,
            situation_key=_situation_key(context),
            step=self.step_index,
        )
        if self.mode == "poker_energy":
            # Confidence is an attribution-only energy/logit analogue. It should
            # not be read as a separate learned poker mechanism.
            strength = min(1.0, len(context.current_opponent_actions) / 2.0)
            p = _sigmoid(model.confidence_logit)
            model.confidence_logit += self.cfg.confidence_lr * strength * (1.0 - p)
        model.confidence_logit *= self.cfg.confidence_decay

        for action in context.current_opponent_actions:
            if action in ACTION_NAMES:
                self.recent_events.append((context.identity_key, context.phase, action))
        while len(self.recent_events) > max(1, self.cfg.sliding_window):
            self.recent_events.popleft()

        self.last_update = {
            "kind": "prompt_actions",
            "identity_key": context.identity_key,
            "actions": list(context.current_opponent_actions),
            "model_count": len(self.models),
            "mode": self.mode,
            "identity_mode": context.identity_mode,
        }

    def update_from_observation(
        self,
        *,
        identity_key: str | None,
        delta: int | None,
        decision: str | None = None,
        phase: str | None = None,
    ) -> None:
        """Absorb end-of-hand reward feedback."""

        if identity_key is None or delta is None:
            return
        self.step_index += 1
        model = self._model_for_key(identity_key)
        model.update_profit(delta, step=self.step_index)
        model.update_decision_reward(
            decision=decision,
            phase=phase,
            delta=delta,
            step=self.step_index,
        )
        if self.mode == "poker_energy":
            # Positive feedback increases confidence; negative feedback lowers it.
            signed = max(-1.0, min(1.0, delta / 100.0))
            model.confidence_logit += self.cfg.confidence_lr * signed
        self.last_update = {
            "kind": "hand_result",
            "identity_key": identity_key,
            "delta": delta,
            "decision": decision,
            "phase": phase,
            "model_count": len(self.models),
            "mode": self.mode,
        }

    def stats_for(self, context: PokerContext) -> dict[str, float]:
        """Return action tendencies for the current decision."""

        mode = self.mode
        if mode in {
            "poker_static_policy",
            "poker_current_hand_only",
            "poker_no_write",
        }:
            return _empty_stats()

        if mode == "poker_sliding_window":
            return self._window_stats(context)

        model = self._find_model(context.identity_key)
        if model is None:
            return _empty_stats()
        return self._model_stats(
            model, phase=context.phase, situation_key=_situation_key(context)
        )

    def artifacts(self) -> dict[str, Any]:
        return {
            "runtime": "poker_online_memory",
            "mode": self.mode,
            "identity_mode": "name_visible" if self.use_opponent_name else "behavior_only",
            "config": asdict(self.cfg),
            "step_index": self.step_index,
            "model_count": len(self.models),
            "last_update": dict(self.last_update),
            "models": [
                {
                    "key": model.key,
                    "created_at": model.created_at,
                    "last_seen": model.last_seen,
                    "actions": dict(model.actions),
                    "hands": model.hands,
                    "total_profit": model.total_profit,
                    "confidence_logit": round(model.confidence_logit, 6),
                    "decision_counts": dict(model.decision_counts),
                    "decision_reward_sum": dict(model.decision_reward_sum),
                    "situation_action_keys": sorted(model.situation_actions),
                    "rates": {
                        action.lower(): round(model.rate(action), 6)
                        for action in ACTION_NAMES
                    },
                }
                for model in self.models
            ],
            "recent_event_count": len(self.recent_events),
        }

    def _window_stats(self, context: PokerContext) -> dict[str, float]:
        counts: Counter[str] = Counter()
        for key, _phase, action in self.recent_events:
            if self.use_opponent_name and key != context.identity_key:
                continue
            counts[action] += 1
        total = sum(counts.values())
        if total == 0:
            return _empty_stats()
        return _stats_from_counts(counts)

    def _model_stats(
        self, model: PokerOpponentModel, *, phase: str, situation_key: str
    ) -> dict[str, float]:
        stats = _stats_from_counts(model.actions)
        street_counts = model.street_actions.get(str(phase).upper())
        if street_counts and sum(street_counts.values()) >= 2:
            street_stats = _stats_from_counts(street_counts)
            # Current-street behavior is more predictive, but keep the global
            # model as shrinkage so early street counts do not overfit.
            street_weight = min(0.70, sum(street_counts.values()) / 8.0)
            for action in ("fold", "call", "check", "raise"):
                key = f"{action}_rate"
                stats[key] = (
                    street_weight * street_stats[key]
                    + (1.0 - street_weight) * stats[key]
                )
            stats["street_action_total"] = street_stats["action_total"]
        else:
            stats["street_action_total"] = 0.0

        situation_counts = model.situation_actions.get(situation_key)
        if situation_counts and sum(situation_counts.values()) >= 2:
            situation_stats = _stats_from_counts(situation_counts)
            situation_weight = min(0.80, sum(situation_counts.values()) / 6.0)
            for action in ("fold", "call", "check", "raise"):
                key = f"{action}_rate"
                stats[key] = (
                    situation_weight * situation_stats[key]
                    + (1.0 - situation_weight) * stats[key]
                )
            stats["situation_action_total"] = situation_stats["action_total"]
        else:
            stats["situation_action_total"] = 0.0

        stats["confidence"] = _sigmoid(model.confidence_logit)
        stats["hands"] = float(model.hands)
        stats["total_profit"] = float(model.total_profit)
        stats["reward_mean"] = (
            float(model.total_profit) / float(model.hands) if model.hands else 0.0
        )
        for action in ACTION_NAMES:
            count = model.decision_counts[action]
            stats[f"reward_{action.lower()}"] = (
                float(model.decision_reward_sum[action]) / float(count)
                if count
                else 0.0
            )
        return stats

    def _find_model(self, key: str) -> PokerOpponentModel | None:
        for model in self.models:
            if model.key == key:
                return model
        return None

    def _model_for_key(self, key: str) -> PokerOpponentModel:
        found = self._find_model(key)
        if found is not None:
            return found
        if len(self.models) >= max(1, self.cfg.capacity):
            self.models.pop(self._weakest_index())
        model = PokerOpponentModel(
            key=key,
            created_at=self.step_index,
            last_seen=self.step_index,
        )
        self.models.append(model)
        return model

    def _weakest_index(self) -> int:
        weakest = 0
        weakest_score = float("inf")
        for idx, model in enumerate(self.models):
            score = model.hands + 0.1 * model.action_total() + model.confidence_logit
            if score < weakest_score:
                weakest = idx
                weakest_score = score
        return weakest


def parse_poker_context(
    *,
    prompt: str,
    metadata: dict[str, Any],
    use_opponent_name: bool,
) -> PokerContext:
    """Parse a Poker query into a compact decision context."""

    legal_actions = [str(a).upper() for a in metadata.get("legal_actions", [])]
    hand_id = str(metadata.get("instance_id") or metadata.get("hand_num") or "unknown")
    hand_num = _as_int(metadata.get("hand_num"), default=0)
    phase = str(metadata.get("phase") or _phase_from_prompt(prompt) or "UNKNOWN")
    opponent_name = str(metadata.get("opponent_name") or _opponent_from_prompt(prompt))
    chips_to_call = _as_int(metadata.get("chips_to_call"), default=0)
    min_raise = _optional_int(metadata.get("min_raise"))
    max_raise = _optional_int(metadata.get("max_raise"))
    pot = _as_int(metadata.get("pot"), default=0)
    player_chips = _as_int(metadata.get("player_chips"), default=0)
    hand_cards = _parse_cards(_line_value(HAND_RE, prompt))
    board_cards = _parse_cards(_line_value(BOARD_RE, prompt))
    board_texture = _board_texture(board_cards)
    strength = estimate_hand_strength(hand_cards, board_cards)
    current_actions = _parse_opponent_actions(prompt)

    if use_opponent_name and opponent_name:
        identity_key = f"name:{opponent_name.lower()}"
        identity_mode = "name_visible"
    else:
        identity_key = "behavior:global"
        identity_mode = "behavior_only"

    return PokerContext(
        hand_id=hand_id,
        hand_num=hand_num,
        phase=phase,
        opponent_name=opponent_name,
        legal_actions=legal_actions,
        chips_to_call=chips_to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        pot=pot,
        player_chips=player_chips,
        board_texture=board_texture,
        strength=strength,
        current_opponent_actions=current_actions,
        identity_key=identity_key,
        identity_mode=identity_mode,
    )


def choose_poker_action(
    *,
    context: PokerContext,
    stats: dict[str, float],
) -> tuple[str, int | None, str]:
    """Choose a legal action by scoring compact action-value candidates.

    The same EV decoder is used by static, sliding-window, vanilla, hard-cache,
    and energy modes.  The modes differ only in the memory statistics they feed
    to this decoder.  This keeps Poker a test of memory quality rather than a
    different hand-coded policy per arm.
    """

    candidates = _candidate_actions(context)
    if not candidates:
        return "CHECK", None, _reason(context, stats, "no legal metadata fallback")

    scored = [
        (action, amount, _action_ev(context, stats, action, amount))
        for action, amount in candidates
    ]
    action_name, amount, ev = max(scored, key=lambda item: item[2])
    return action_name, amount, _reason(
        context,
        stats,
        f"ev_decision={action_name}; ev={ev:.2f}; candidates={_format_scores(scored)}",
    )


def parse_hand_delta(observation_content: str) -> int | None:
    match = NET_RE.search(observation_content)
    if match is None:
        return None
    return _as_int(match.group("delta"), default=0)


def _candidate_actions(context: PokerContext) -> list[tuple[str, int | None]]:
    legal = set(context.legal_actions)
    candidates: list[tuple[str, int | None]] = []
    for action in ("FOLD", "CALL", "CHECK"):
        if action in legal:
            candidates.append((action, None))
    if "RAISE" in legal:
        for amount in _raise_amounts(context):
            candidates.append(("RAISE", amount))
    return candidates


def _raise_amounts(context: PokerContext) -> list[int]:
    if context.min_raise is None:
        return []
    min_total = max(0, int(context.min_raise))
    max_total = int(context.max_raise) if context.max_raise is not None else min_total
    if max_total < min_total:
        return []

    totals = {min_total}
    pot = max(1, context.pot)
    # These are total raise targets, not additive bet sizes.  They deliberately
    # include more than the minimum so the policy can stop the min-raise wars
    # observed in the negative Poker trace.
    for fraction in (0.50, 0.75, 1.00, 1.50):
        total = int(round(max(min_total, min(max_total, pot * fraction))))
        totals.add(total)
    # Include a stack-pressure size only with premium equity.  The original
    # negative trace cratered through raise wars and stack-offs; a generic EV
    # decoder should have a risk gate, not blindly expose all-in as a default
    # candidate.
    if context.strength >= 0.82:
        totals.add(max_total)
    return sorted(total for total in totals if min_total <= total <= max_total)


def _action_ev(
    context: PokerContext,
    stats: dict[str, float],
    action: str,
    amount: int | None,
) -> float:
    legal = set(context.legal_actions)
    equity = max(0.01, min(0.99, context.strength))
    pot = float(max(0, context.pot))
    to_call = float(max(0, context.chips_to_call))
    facing_bet = "CALL" in legal and "CHECK" not in legal

    action_total = stats.get("action_total", 0.0)
    confidence = stats.get("confidence", 0.0)
    reliability = max(0.0, min(1.0, 0.25 * confidence + 0.75 * min(1.0, action_total / 10.0)))

    fold_rate = _blend(0.25, stats.get("fold_rate", 0.25), reliability)
    call_rate = _blend(0.30, stats.get("call_rate", 0.25), reliability)
    check_rate = _blend(0.25, stats.get("check_rate", 0.25), reliability)
    raise_rate = _blend(0.20, stats.get("raise_rate", 0.25), reliability)
    aggression = max(0.0, min(1.0, raise_rate + 0.5 * max(0.0, raise_rate - fold_rate)))

    # Incremental EV from the current decision.  FOLD avoids further loss, so it
    # is the zero reference.  This makes weak calls fold naturally.
    if action == "FOLD":
        if not facing_bet:
            return -1.0
        return 0.0

    if action == "CHECK":
        if "CHECK" not in legal:
            return float("-inf")
        showdown = (2.0 * equity - 1.0) * pot * 0.35
        # Checking to an aggressive player can surrender initiative, but it is
        # still preferable to bloating the pot with marginal equity.
        return showdown - 0.08 * aggression * pot

    if action == "CALL":
        if "CALL" not in legal:
            return float("-inf")
        call_ev = equity * pot - (1.0 - equity) * to_call
        # LAG pressure was the observed crater.  Penalize marginal calls into a
        # high-raise opponent and reward calls against call-heavy/passive lines.
        call_ev -= max(0.0, 0.55 - equity) * aggression * max(to_call, pot * 0.25)
        call_ev += max(0.0, call_rate - raise_rate) * equity * min(pot, to_call + 1.0)
        return call_ev + _reward_bonus(stats, action, reliability)

    if action == "RAISE":
        if amount is None or "RAISE" not in legal:
            return float("-inf")
        risk = float(max(0, amount))
        if context.max_raise is not None:
            risk = min(risk, float(max(0, context.max_raise)))

        continue_rate = max(0.05, min(0.95, call_rate + 0.65 * raise_rate))
        # If the opponent is aggressive, a raise is more likely to be met with
        # more pressure than with an immediate fold.
        effective_fold = max(0.02, min(0.90, fold_rate * (1.0 - 0.45 * aggression)))
        effective_continue = max(0.05, min(0.98, continue_rate))
        norm = effective_fold + effective_continue
        effective_fold /= norm
        effective_continue /= norm

        called_ev = equity * (pot + risk) - (1.0 - equity) * risk
        fold_ev = pot
        raise_ev = effective_fold * fold_ev + effective_continue * called_ev

        # Avoid the exact failure observed in the trace: marginal min-raise wars
        # against LAG.  Strong hands can still value raise.
        if aggression >= 0.38 and equity < 0.68:
            raise_ev -= (0.68 - equity) * risk * (0.75 + aggression)
        if risk <= max(1.0, float(context.min_raise or 0)) and aggression >= 0.30:
            raise_ev -= 0.20 * risk

        # Bet big only when either value is strong or fold equity is real.
        pot_fraction = risk / max(1.0, pot)
        stack_fraction = risk / max(1.0, float(context.player_chips or risk or 1.0))
        if pot_fraction > 1.20 and equity < 0.72 and effective_fold < 0.45:
            raise_ev -= (pot_fraction - 1.20) * risk
        if stack_fraction > 0.45 and equity < 0.82:
            raise_ev -= (stack_fraction - 0.45) * risk * 1.5

        return raise_ev + _reward_bonus(stats, action, reliability)

    return float("-inf")


def _reward_bonus(stats: dict[str, float], action: str, reliability: float) -> float:
    # Reward-prediction feedback is deliberately a small correction.  It should
    # shape choices after repeated evidence without becoming a bespoke score
    # table that overwhelms hand equity and pot odds.
    return 0.15 * reliability * stats.get(f"reward_{action.lower()}", 0.0)


def _blend(prior: float, observed: float, weight: float) -> float:
    weight = max(0.0, min(1.0, weight))
    return (1.0 - weight) * prior + weight * observed


def _format_scores(scored: list[tuple[str, int | None, float]]) -> str:
    compact = []
    for action, amount, ev in scored:
        label = action if amount is None else f"{action}{amount}"
        compact.append(f"{label}:{ev:.1f}")
    return ",".join(compact[:8])


def _situation_key(context: PokerContext) -> str:
    legal = set(context.legal_actions)
    facing = "facing_bet" if "CALL" in legal and "CHECK" not in legal else "no_bet"
    last_action = (
        context.current_opponent_actions[-1]
        if context.current_opponent_actions
        else "NONE"
    )
    pressure = "opp_raised" if "RAISE" in context.current_opponent_actions else "no_raise"
    to_call_bucket = _bucket(context.chips_to_call, [(0, "free"), (20, "small"), (80, "mid")], "large")
    return "|".join(
        [
            str(context.phase).upper(),
            facing,
            pressure,
            context.board_texture,
            to_call_bucket,
            last_action,
        ]
    )


def _bucket(value: int, thresholds: list[tuple[int, str]], default: str) -> str:
    for threshold, label in thresholds:
        if value <= threshold:
            return label
    return default


def _board_texture(board_cards: list[tuple[int, str]]) -> str:
    if not board_cards:
        return "preflop"
    ranks = [rank for rank, _suit in board_cards]
    suits = [suit for _rank, suit in board_cards]
    paired = len(set(ranks)) < len(ranks)
    high = max(ranks)
    suit_counts = Counter(suits)
    flushy = max(suit_counts.values(), default=0) >= min(3, len(board_cards))
    sorted_ranks = sorted(set(ranks))
    connected = any(
        sorted_ranks[idx + 1] - sorted_ranks[idx] <= 2
        for idx in range(len(sorted_ranks) - 1)
    )
    parts = []
    if high >= 13:
        parts.append("high")
    else:
        parts.append("low")
    if paired:
        parts.append("paired")
    if flushy:
        parts.append("flushy")
    if connected:
        parts.append("connected")
    return "_".join(parts)


def estimate_hand_strength(
    hand_cards: list[tuple[int, str]], board_cards: list[tuple[int, str]]
) -> float:
    """Cheap, monotonic hand-strength proxy in [0, 1]."""

    if len(hand_cards) < 2:
        return 0.45

    treys_equity = _treys_equity(hand_cards, board_cards)
    if treys_equity is not None:
        return treys_equity

    ranks = [rank for rank, _suit in hand_cards]
    suits = [suit for _rank, suit in hand_cards]
    high = max(ranks)
    low = min(ranks)
    suited = len(set(suits)) == 1
    pair = ranks[0] == ranks[1]
    connected = abs(ranks[0] - ranks[1]) <= 1

    if not board_cards:
        strength = 0.20 + 0.025 * (high + low)
        if pair:
            strength += 0.22 + 0.012 * high
        if suited:
            strength += 0.05
        if connected:
            strength += 0.04
        if high >= 13:
            strength += 0.05
        return max(0.05, min(0.95, strength))

    combined = hand_cards + board_cards
    counts = Counter(rank for rank, _suit in combined)
    max_count = max(counts.values())
    board_high = max(rank for rank, _suit in board_cards)
    hole_pair = any(counts[rank] >= 2 for rank in ranks)
    two_pairish = sum(1 for count in counts.values() if count >= 2) >= 2
    flush_draw = any(
        sum(1 for _rank, suit in combined if suit == candidate_suit) >= 4
        for candidate_suit in {suit for _rank, suit in combined}
    )

    strength = 0.25 + 0.02 * high
    if max_count >= 4:
        strength = 0.95
    elif max_count == 3:
        strength = 0.82
    elif two_pairish:
        strength = 0.72
    elif hole_pair:
        strength = 0.58 + (0.08 if high >= board_high else 0.0)
    elif flush_draw:
        strength = 0.48
    return max(0.05, min(0.98, strength))


def _treys_equity(
    hand_cards: list[tuple[int, str]], board_cards: list[tuple[int, str]]
) -> float | None:
    if TREYS_EVALUATOR is None or TreysCard is None:
        return None
    hand_key = tuple(_card_key(card) for card in hand_cards)
    board_key = tuple(_card_key(card) for card in board_cards)
    if any(card is None for card in hand_key) or any(card is None for card in board_key):
        return None
    return _treys_equity_cached(
        tuple(sorted(card for card in hand_key if card is not None)),
        tuple(sorted(card for card in board_key if card is not None)),
    )


@lru_cache(maxsize=8192)
def _treys_equity_cached(hand_key: tuple[str, ...], board_key: tuple[str, ...]) -> float:
    if TREYS_EVALUATOR is None or TreysCard is None:
        return 0.50
    if len(hand_key) != 2:
        return 0.45
    if len(board_key) == 0:
        return _preflop_equity_from_key(hand_key)

    hole = [TreysCard.new(card) for card in hand_key]
    board = [TreysCard.new(card) for card in board_key]
    known = set(hole + board)
    deck = [card for card in TREYS_FULL_DECK if card not in known]
    unknown_board = max(0, 5 - len(board))

    seed_text = "|".join([*hand_key, *board_key])
    rng = _deterministic_rng(seed_text)
    samples = 96 if unknown_board >= 2 else 192
    wins = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, 2 + unknown_board)
        opp_hole = draw[:2]
        runout = draw[2:]
        final_board = board + runout
        our_rank = TREYS_EVALUATOR.evaluate(hole, final_board)
        opp_rank = TREYS_EVALUATOR.evaluate(opp_hole, final_board)
        if our_rank < opp_rank:
            wins += 1.0
        elif our_rank == opp_rank:
            wins += 0.5
    return max(0.01, min(0.99, wins / float(samples)))


def _preflop_equity_from_key(hand_key: tuple[str, ...]) -> float:
    ranks = sorted((RANK_VALUE[card[:-1]] for card in hand_key), reverse=True)
    suited = hand_key[0][-1] == hand_key[1][-1]
    pair = ranks[0] == ranks[1]
    connected = abs(ranks[0] - ranks[1]) <= 1
    equity = 0.30 + 0.018 * (ranks[0] + ranks[1])
    if pair:
        equity += 0.16 + ranks[0] / 100.0
    if suited:
        equity += 0.035
    if connected:
        equity += 0.025
    if ranks[0] == 14:
        equity += 0.04
    return max(0.05, min(0.95, equity))


def _card_key(card: tuple[int, str]) -> str | None:
    rank, suit = card
    rank_text = {
        14: "A",
        13: "K",
        12: "Q",
        11: "J",
        10: "T",
    }.get(rank, str(rank))
    suit_text = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}.get(suit, suit)
    if suit_text not in {"s", "h", "d", "c"}:
        return None
    return f"{rank_text}{suit_text}"


def _deterministic_rng(seed_text: str):
    # Local import keeps module import cheap in system validation.
    import random

    return random.Random(seed_text)


def _stats_from_counts(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    smoothed_total = total + len(ACTION_NAMES)
    return {
        "fold_rate": (counts["FOLD"] + 1.0) / smoothed_total,
        "call_rate": (counts["CALL"] + 1.0) / smoothed_total,
        "check_rate": (counts["CHECK"] + 1.0) / smoothed_total,
        "raise_rate": (counts["RAISE"] + 1.0) / smoothed_total,
        "action_total": float(total),
        "confidence": min(1.0, total / 12.0),
    }


def _empty_stats() -> dict[str, float]:
    stats = {
        "fold_rate": 0.25,
        "call_rate": 0.25,
        "check_rate": 0.25,
        "raise_rate": 0.25,
        "action_total": 0.0,
        "street_action_total": 0.0,
        "situation_action_total": 0.0,
        "confidence": 0.0,
        "hands": 0.0,
        "total_profit": 0.0,
        "reward_mean": 0.0,
    }
    for action in ACTION_NAMES:
        stats[f"reward_{action.lower()}"] = 0.0
    return stats


def _parse_opponent_actions(prompt: str) -> list[str]:
    match = OPP_ACTION_RE.search(prompt)
    if match is None:
        return []
    raw = match.group("actions").upper()
    return [
        token
        for token in re.findall(r"\b(FOLD|CALL|CHECK|RAISE)\b", raw)
        if token in ACTION_NAMES
    ]


def _parse_cards(raw: str | None) -> list[tuple[int, str]]:
    if not raw or "No cards" in raw:
        return []
    cards: list[tuple[int, str]] = []
    for token in raw.replace(",", " ").split():
        cleaned = token.strip("[](){}")
        rank_match = re.search(r"(10|[2-9TJQKA])", cleaned, re.IGNORECASE)
        if rank_match is None:
            continue
        rank = RANK_VALUE.get(rank_match.group(1).upper())
        if rank is None:
            continue
        suit = "?"
        for candidate in ("♠", "♥", "♦", "♣", "s", "h", "d", "c"):
            if candidate in cleaned:
                suit = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}.get(
                    candidate, candidate.lower()
                )
                break
        cards.append((rank, suit))
    return cards


def _line_value(pattern: re.Pattern[str], prompt: str) -> str | None:
    match = pattern.search(prompt)
    if match is None:
        return None
    return match.group("cards").strip()


def _phase_from_prompt(prompt: str) -> str | None:
    match = re.search(r"Hand\s+#\d+\s+-\s+(?P<phase>[A-Z_]+)", prompt)
    if match is None:
        return None
    return match.group("phase")


def _opponent_from_prompt(prompt: str) -> str:
    match = re.search(r"Opponent:\s*(?P<name>.+)", prompt)
    if match is None:
        return "Opponent"
    return match.group("name").strip()


def _safe_raise_amount(context: PokerContext) -> int | None:
    if context.min_raise is None:
        return None
    amount = context.min_raise
    if context.max_raise is not None:
        amount = min(amount, context.max_raise)
    return max(0, int(amount))


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _reason(
    context: PokerContext, stats: dict[str, float], decision: str
) -> str:
    return (
        f"{decision}; strength={context.strength:.2f}, "
        f"fold={stats.get('fold_rate', 0.25):.2f}, "
        f"call={stats.get('call_rate', 0.25):.2f}, "
        f"raise={stats.get('raise_rate', 0.25):.2f}, "
        f"identity={context.identity_mode}"
    )
