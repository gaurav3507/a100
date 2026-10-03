"""Pydantic schemas for LLM output, and the JSON schemas handed to vLLM.

Design note: every string field carries a hard `max_length`. That length lands
in the JSON Schema as `maxLength`, which guided decoding enforces at generation
time. Without it the model simply pastes whole sentences out of the transcript,
which produces a transcription rather than an analysis. The limits are the
mechanism, not a suggestion.

The only place verbatim text is wanted is `notable_quotes`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TONES = ("confident", "cautious", "defensive", "neutral")


class SegmentSplit(BaseModel):
    name: str = Field(max_length=40, description="Segment name only")
    revenue_share: str = Field(
        default="n/a", max_length=40,
        description="Share or revenue, e.g. '40% of rev' or 'INR393cr'. 'n/a' if unstated.",
    )


class TranscriptAnalysis(BaseModel):
    business_summary: str = Field(
        max_length=160,
        description="What the company does, one compressed line. No preamble.",
    )
    segments: list[SegmentSplit] = Field(default_factory=list, max_length=6)
    key_numbers: list[str] = Field(
        default_factory=list, max_length=6,
        description="Headline metrics, telegraphic. e.g. 'Rev 251cr +27% YoY', "
                    "'EBITDA margin 26.5%, +310bps'. One metric per item.",
    )
    what_changed: list[str] = Field(
        default_factory=list, max_length=5,
        description="What is DIFFERENT this quarter versus prior quarters. "
                    "Only genuine changes, not routine performance. Empty list "
                    "if nothing changed.",
    )
    new_themes: list[str] = Field(
        default_factory=list, max_length=6,
        description="New business area, technology, customer or end-market. "
                    "Short noun phrases only, e.g. 'semiconductor trays'.",
    )
    order_book: str = Field(default="n/a", max_length=120)
    capex: str = Field(default="n/a", max_length=120)
    guidance: str = Field(default="n/a", max_length=140)
    margin_commentary: str = Field(
        default="n/a", max_length=140,
        description="WHY margins moved, not what they were. Cause, not restatement.",
    )
    demand_commentary: str = Field(
        default="n/a", max_length=140,
        description="State of end-market demand, compressed.",
    )
    management_tone: Literal["confident", "cautious", "defensive", "neutral"] = "neutral"
    tone_justification: str = Field(
        default="", max_length=120,
        description="One clause on why, ideally grounded in the Q&A.",
    )
    notable_quotes: list[str] = Field(
        default_factory=list, max_length=3,
        description="Verbatim quotes, the ONLY field where copying is correct. "
                    "Pick lines that carry information, not platitudes.",
    )
    risks_flagged: list[str] = Field(
        default_factory=list, max_length=6,
        description="Concerns raised by management or pressed by analysts. "
                    "Short phrases, not sentences.",
    )


class QuarterDiff(BaseModel):
    narrative_shift: str = Field(
        max_length=400,
        description="Has the story changed across these quarters? What, and from "
                    "which quarter? Be specific and brief. If nothing changed, "
                    "say so plainly.",
    )
    new_in_latest: list[str] = Field(default_factory=list, max_length=6)
    dropped: list[str] = Field(default_factory=list, max_length=6)
    tone_trajectory: str = Field(
        max_length=160,
        description="How tone moved, e.g. 'cautious -> confident from Q2 as "
                    "healthcare scaled'.",
    )
    shift_score: int = Field(ge=0, le=10)


def json_schema(model: type[BaseModel]) -> dict:
    """vLLM guided_json expects a plain JSON Schema dict."""
    return model.model_json_schema()


EMPTY_ANALYSIS = {
    "business_summary": "",
    "segments": [],
    "key_numbers": [],
    "what_changed": [],
    "new_themes": [],
    "order_book": "n/a",
    "capex": "n/a",
    "guidance": "n/a",
    "margin_commentary": "n/a",
    "demand_commentary": "n/a",
    "management_tone": "neutral",
    "tone_justification": "",
    "notable_quotes": [],
    "risks_flagged": [],
}

INSUFFICIENT_DIFF = {
    "narrative_shift": "INSUFFICIENT DATA",
    "new_in_latest": [],
    "dropped": [],
    "tone_trajectory": "INSUFFICIENT DATA",
    "shift_score": 0,
}
