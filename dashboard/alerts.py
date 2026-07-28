"""Configurable alert rules + Microsoft Teams delivery.

Rules are stored as JSON in ``config.RULES_FILE`` and are editable from the
dashboard UI. After each inspection the enabled rules are evaluated for the
affected location; if a rule fires (and is off cooldown) a Teams message is
posted and the alert is logged.

Supported rule types
--------------------
- ``count``      : >= ``threshold`` REJECT panels from a location within
                   ``window_minutes``.
- ``rate``       : reject RATE >= ``threshold_pct`` over ``window_minutes``,
                   requiring at least ``min_samples`` inspections.
- ``immediate``  : any single REJECT triggers immediately.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import requests

import config
import store

DEFAULT_RULES: list[dict] = [
    {
        "name": "Burst of defects",
        "type": "count",
        "enabled": True,
        "threshold": 3,
        "window_minutes": 30,
        "cooldown_minutes": 15,
    },
    {
        "name": "High defect rate",
        "type": "rate",
        "enabled": True,
        "threshold_pct": 25.0,
        "window_minutes": 60,
        "min_samples": 8,
        "cooldown_minutes": 30,
    },
    {
        "name": "Immediate broken chip",
        "type": "immediate",
        "enabled": False,
        "cooldown_minutes": 5,
    },
]

# In-memory cooldown tracker: (rule_name, location) -> last_fired datetime.
_last_fired: dict[tuple[str, str], datetime] = {}


def load_rules() -> list[dict]:
    if config.RULES_FILE.exists():
        try:
            return json.loads(config.RULES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    save_rules(DEFAULT_RULES)
    return list(DEFAULT_RULES)


def save_rules(rules: list[dict]) -> None:
    config.RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.RULES_FILE.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def _on_cooldown(rule: dict, location: str) -> bool:
    last = _last_fired.get((rule["name"], location))
    if last is None:
        return False
    cooldown = timedelta(minutes=rule.get("cooldown_minutes", 0))
    return datetime.now() - last < cooldown


def _evaluate_rule(rule: dict, location: str) -> str | None:
    """Return an alert message if the rule fires, else None."""
    rtype = rule.get("type")
    if rtype == "immediate":
        return f"Broken chip detected at **{location}**."
    if rtype == "count":
        window = int(rule.get("window_minutes", 30))
        n = store.recent_rejects(location, window)
        if n >= int(rule.get("threshold", 3)):
            return (
                f"**{n}** rejected panels at **{location}** in the last "
                f"{window} min (threshold {rule.get('threshold')})."
            )
    elif rtype == "rate":
        window = int(rule.get("window_minutes", 60))
        rejects, total = store.window_stats(location, window)
        min_samples = int(rule.get("min_samples", 8))
        if total >= min_samples:
            pct = 100.0 * rejects / total if total else 0.0
            if pct >= float(rule.get("threshold_pct", 25.0)):
                return (
                    f"Defect rate **{pct:.0f}%** at **{location}** "
                    f"({rejects}/{total}) over last {window} min "
                    f"(threshold {rule.get('threshold_pct')}%)."
                )
    return None


def _is_workflow_webhook(url: str) -> bool:
    """True for Power Automate / Teams "Workflows" webhooks.

    These replace the retiring O365 Incoming Webhook connector and expect an
    Adaptive Card wrapped in a ``{type:"message", attachments:[...]}`` envelope
    rather than the legacy ``MessageCard`` schema.
    """
    u = url.lower()
    return (
        "powerplatform.com" in u
        or "powerautomate" in u
        or "logic.azure.com" in u
        or "/workflows/" in u
    )


def _adaptive_card_payload(title: str, message: str, location: str | None) -> dict:
    """Build the Adaptive Card envelope for Power Automate / Workflows webhooks."""
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": f"🚨 {title}",
                            "weight": "Bolder",
                            "size": "Large",
                            "color": "Attention",
                            "wrap": True,
                        },
                        {
                            "type": "TextBlock",
                            "text": "FS50 Corner Metrology",
                            "isSubtle": True,
                            "spacing": "None",
                            "wrap": True,
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "Location", "value": location or "-"},
                                {
                                    "title": "Time",
                                    "value": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                },
                            ],
                        },
                        {"type": "TextBlock", "text": message, "wrap": True},
                    ],
                },
            }
        ],
    }


def _message_card_payload(title: str, message: str, location: str | None) -> dict:
    """Legacy O365 connector (outlook.office.com) MessageCard payload."""
    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "themeColor": "D93025",
        "summary": title,
        "sections": [
            {
                "activityTitle": f"🚨 {title}",
                "activitySubtitle": "FS50 Corner Metrology",
                "facts": [
                    {"name": "Location", "value": location or "-"},
                    {
                        "name": "Time",
                        "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                ],
                "text": message,
                "markdown": True,
            }
        ],
    }


def post_to_teams(title: str, message: str, location: str | None = None) -> bool:
    """Send an alert to Teams. Returns delivery success.

    Supports both the new Power Automate / Teams "Workflows" webhooks (Adaptive
    Card envelope) and the legacy O365 Incoming Webhook connector (MessageCard),
    auto-detected from the webhook URL.
    """
    url = config.TEAMS_WEBHOOK_URL
    if not url:
        return False
    if _is_workflow_webhook(url):
        payload = _adaptive_card_payload(title, message, location)
    else:
        payload = _message_card_payload(title, message, location)
    try:
        resp = requests.post(url, json=payload, timeout=10)
        # Workflow webhooks return 202 Accepted; connectors return 200.
        return resp.status_code < 300
    except Exception:
        return False


def check_and_alert(location: str) -> list[dict]:
    """Evaluate all enabled rules for a location; fire + log any that trigger."""
    fired: list[dict] = []
    for rule in load_rules():
        if not rule.get("enabled"):
            continue
        if _on_cooldown(rule, location):
            continue
        message = _evaluate_rule(rule, location)
        if not message:
            continue
        delivered = post_to_teams(rule["name"], message, location)
        store.log_alert(
            ts=datetime.now(),
            rule_name=rule["name"],
            location=location,
            message=message,
            delivered=delivered,
        )
        _last_fired[(rule["name"], location)] = datetime.now()
        fired.append({"rule": rule["name"], "message": message, "delivered": delivered})
    return fired


# --------------------------------------------------------------------------
# Metrology (edge-grind) + FS100-broken Teams cadence
# --------------------------------------------------------------------------
def _max_in_rolling_window(times: list[datetime], window: timedelta) -> int:
    """Largest number of timestamps falling inside any ``window``-long span."""
    ts = sorted(t for t in times if t is not None)
    best = 0
    j = 0
    for i in range(len(ts)):
        while ts[i] - ts[j] > window:
            j += 1
        best = max(best, i - j + 1)
    return best


def _commonality_phrase(df, col: str, label: str) -> str | None:
    """Summarise the dominant values of a column, e.g. ``side: Left×3, Right×1``."""
    if col not in df.columns:
        return None
    vc = df[col].dropna().value_counts()
    if vc.empty:
        return None
    parts = [f"{k}×{int(v)}" for k, v in vc.items()]
    return f"{label}: " + ", ".join(parts)


def send_defect_burst_alerts(
    alerts_df,
    window_minutes: int = 15,
    min_count: int = 3,
    cooldown_minutes: int = 15,
) -> list[dict]:
    """Fire a Teams alert when a grinder logs >= ``min_count`` edge-grind defects
    within ``window_minutes``.

    ``alerts_df`` is the run-length defect frame from
    :func:`metrology.build_alerts` (columns ``read_time``, ``grinder``,
    ``defect``, ``side``, ``edge``, ``groove``, ``length_mm``). One alert per
    grinder per cooldown; the message summarises the commonality (defect types,
    sides, edges, grooves) so an engineer sees the pattern at a glance.
    """
    fired: list[dict] = []
    if alerts_df is None or getattr(alerts_df, "empty", True):
        return fired
    window = timedelta(minutes=window_minutes)
    for grinder, grp in alerts_df.groupby("grinder"):
        if not grinder or grinder in ("?", None):
            continue
        times = list(grp["read_time"])
        peak = _max_in_rolling_window(times, window)
        if peak < min_count:
            continue
        location = f"FS50/{grinder}"
        rule_name = "FS50 defect burst"
        if _on_cooldown(
            {"name": rule_name, "cooldown_minutes": cooldown_minutes}, location
        ):
            continue
        bits = [
            f"**{int(peak)}** defects within {window_minutes} min on **{grinder}** "
            f"({len(grp)} total in the current window).",
        ]
        for col, lbl in (
            ("defect", "type"),
            ("side", "side"),
            ("edge", "edge"),
            ("groove", "groove"),
        ):
            phrase = _commonality_phrase(grp, col, lbl)
            if phrase:
                bits.append(phrase)
        message = "  \n".join(bits)
        delivered = post_to_teams(rule_name, message, grinder)
        store.log_alert(
            ts=datetime.now(),
            rule_name=rule_name,
            location=grinder,
            message=message,
            delivered=delivered,
        )
        _last_fired[(rule_name, location)] = datetime.now()
        fired.append({"rule": rule_name, "message": message, "delivered": delivered})
    return fired


def send_broken_alerts(broken_df, cooldown_minutes: int = 15) -> list[dict]:
    """Fire a Teams alert for FS100-broken plates (post-VTD scrap).

    ``broken_df`` is the backtrack frame from
    :func:`metrology.broken_with_attribution` (columns ``SubID``, ``VtdName``,
    ``BrokenTime``, ``grinder``, ``edge``, ``groove``, ``defect``,
    ``had_defect``). Each broken SubID alerts once (cooldown-keyed by SubID) and
    the message backtracks to the grinder/groove/edge so the plant can see
    *why* it likely broke.
    """
    fired: list[dict] = []
    if broken_df is None or getattr(broken_df, "empty", True):
        return fired
    rule_name = "FS100 broken"
    for r in broken_df.itertuples():
        sid = str(getattr(r, "SubID", ""))
        location = f"FS100/{sid}"
        if _on_cooldown(
            {"name": rule_name, "cooldown_minutes": cooldown_minutes}, location
        ):
            continue
        grinder = getattr(r, "grinder", None) or "unknown grinder"
        vtd = getattr(r, "VtdName", None) or "a VTD line"
        bits = [f"Plate **{sid}** broke at **{vtd}** (FS100)."]
        trace = []
        if getattr(r, "grinder", None):
            trace.append(f"ground on {grinder}")
        if getattr(r, "edge", None):
            trace.append(f"{r.edge} edge")
        if getattr(r, "groove", None):
            trace.append(f"groove {r.groove}")
        if trace:
            bits.append("Lineage: " + ", ".join(trace) + ".")
        if getattr(r, "had_defect", False) and getattr(r, "defect", None):
            side = getattr(r, "side", None)
            bits.append(
                f"⚠️ Edge-grind defect was already detected: {r.defect}"
                + (f" ({side} side)" if side else "")
                + " — likely root cause."
            )
        else:
            bits.append(
                "No prior edge-grind defect was flagged — a new failure mode to learn from."
            )
        message = "  \n".join(bits)
        delivered = post_to_teams(rule_name, message, grinder)
        store.log_alert(
            ts=datetime.now(),
            rule_name=rule_name,
            location=grinder,
            message=message,
            delivered=delivered,
        )
        _last_fired[(rule_name, location)] = datetime.now()
        fired.append(
            {
                "rule": rule_name,
                "message": message,
                "delivered": delivered,
                "sub_id": sid,
            }
        )
    return fired
