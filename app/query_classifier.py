"""
Query classifier — routes a user query to the correct manual collection.

Returns one of:
  "parts"       → search parts_manuals  (hybrid BM25 + vector)
  "maintenance" → search maintenance_manuals  (vector-only)
  "both"        → search both collections and merge

Rule-based, no LLM call.
"""

import re
from typing import Literal
from app.retriever import tokenize

QueryType = Literal["parts", "maintenance", "both"]

_PART_NUM_RE = re.compile(r"\b\d{3}-\d{4}\b")

# Strong parts signals
_PARTS_TOKENS = {
    "part", "parts", "number", "serial", "item", "quantity", "qty",
    "catalogue", "catalog", "kit", "assembly", "plug", "filter",
    "seal", "bearing", "belt", "hose", "gasket", "valve", "pump",
    "sensor", "relay", "fuse", "alternator", "starter", "injector",
    "nozzle", "ring", "piston", "bolt", "nut", "bracket", "fitting",
    "o-ring", "oring", "bushing", "shim", "washer", "clip", "pin",
}

# Strong maintenance signals (include common -ing/-ed/-tion forms — no stemmer needed)
_MAINTENANCE_TOKENS = {
    "interval", "schedule", "procedure", "torque", "specification",
    "spec", "replace", "replacement", "replacing", "replaced",
    "inspect", "inspection", "inspecting", "check", "checking",
    "change", "changing", "flush", "flushing", "drain", "draining",
    "fill", "filling", "lubrication", "lube", "lubricating",
    "service", "servicing", "maintenance", "overhaul", "overhauling",
    "calibrate", "calibration", "calibrating",
    "adjust", "adjustment", "adjusting", "bleed", "bleeding",
    "prime", "priming", "clean", "cleaning",
    "test", "testing", "troubleshoot", "troubleshooting",
    "diagnosis", "diagnostic", "diagnosing",
    "pressure", "temperature", "clearance", "wear", "limit",
    "capacity", "volume", "viscosity", "grade", "hours", "km", "miles",
}


def classify_query(query: str) -> QueryType:
    """
    Classify a query as 'parts', 'maintenance', or 'both'.

    Decision logic:
    1. Explicit part number (NNN-NNNN) → parts
    2. Count token overlap with parts and maintenance signal sets
    3. If both sides have hits → both
    4. Whichever side has more hits wins
    5. Default to 'maintenance' when no signal is found
    """
    toks = set(tokenize(query))

    # Hard rule: explicit part number → parts
    if _PART_NUM_RE.search(query):
        return "parts"

    parts_hits = toks.intersection(_PARTS_TOKENS)
    maint_hits = toks.intersection(_MAINTENANCE_TOKENS)

    parts_score = len(parts_hits)
    maint_score = len(maint_hits)

    # Strong parts indicators that override weak maintenance overlap
    strong_parts = {"part", "parts", "number", "serial", "catalogue", "catalog", "kit", "assembly"}
    if toks.intersection(strong_parts):
        parts_score += 2

    # Strong maintenance indicators
    strong_maint = {"interval", "procedure", "schedule", "overhaul", "lubrication", "torque"}
    if toks.intersection(strong_maint):
        maint_score += 2

    if parts_score > 0 and maint_score > 0:
        # Both sides have signals — search both unless one dominates
        if parts_score >= maint_score * 2:
            return "parts"
        if maint_score >= parts_score * 2:
            return "maintenance"
        return "both"

    if parts_score > 0:
        return "parts"
    if maint_score > 0:
        return "maintenance"

    # No signal — bias toward maintenance to reduce noisy cross-collection retrieval.
    return "maintenance"


def collection_for_type(query_type: QueryType) -> list[str]:
    """Return the collection name(s) to search for a given query type."""
    if query_type == "parts":
        return ["parts_manuals"]
    if query_type == "maintenance":
        return ["maintenance_manuals"]
    return ["parts_manuals", "maintenance_manuals"]
