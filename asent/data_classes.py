from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel
from spacy.tokens import Doc, Span, Token


class TokenPolarityOutput(BaseModel):
    """A data class for the polarity output of a span, notably allows for
    plotting the output."""

    class Config:
        arbitrary_types_allowed = True

    polarity: float
    token: Token
    span: Span
    negation: Union[Token, None] = None
    intensifiers: List[Token] = []

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}"
            for a, v in [
                ("polarity", round(self.polarity, 3)),
                ("token", self.token),
                ("span", self.span),
            ]
        )

    def __lt__(self, other: object):
        if not isinstance(other, (TokenPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, TokenPolarityOutput):
            other = other.polarity
        return self.polarity < other

    def __gt__(self, other: object):
        if not isinstance(other, (TokenPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, TokenPolarityOutput):
            other = other.polarity
        return self.polarity > other

    def __bool__(self):
        return bool(self.polarity)

    def __eq__(self, other: object):
        if not isinstance(other, (TokenPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, TokenPolarityOutput):
            other = other.polarity
        return self.polarity == other


class SpanPolarityOutput(BaseModel):
    """A data class for the polarity output of a span, notably allows for
    plotting the output."""

    class Config:
        arbitrary_types_allowed = True

    negative: float
    neutral: float
    positive: float
    compound: float
    span: Span
    polarities: List[TokenPolarityOutput]

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}"
            for a, v in [
                ("neg", round(self.negative, 3)),
                ("neu", round(self.neutral, 3)),
                ("pos", round(self.positive, 3)),
                ("compound", round(self.compound, 4)),
                ("span", self.span),
            ]
        )

    def __lt__(self, other: object):
        if not isinstance(other, (SpanPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, SpanPolarityOutput):
            other = other.compound
        return self.compound < other

    def __gt__(self, other: object):
        if not isinstance(other, (SpanPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, SpanPolarityOutput):
            other = other.compound
        return self.compound > other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (SpanPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, SpanPolarityOutput):
            other = other.compound
        return self.compound == other


class DocPolarityOutput(BaseModel):
    """A data class for the polarity output of a doc."""

    class Config:
        arbitrary_types_allowed = True

    negative: float
    neutral: float
    positive: float
    compound: float
    doc: Doc
    polarities: List[SpanPolarityOutput]

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(
            repr(v) if a is None else f"{a}={v!r}"
            for a, v in [
                ("neg", round(self.negative, 3)),
                ("neu", round(self.neutral, 3)),
                ("pos", round(self.positive, 3)),
                ("compound", round(self.compound, 4)),
            ]
        )

    def __lt__(self, other: object):
        if not isinstance(other, (DocPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, DocPolarityOutput):
            other = other.compound
        return self.compound < other

    def __gt__(self, other: object):
        if not isinstance(other, (DocPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, DocPolarityOutput):
            other = other.compound
        return self.compound > other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (DocPolarityOutput, float)):
            return NotImplemented
        if isinstance(other, DocPolarityOutput):
            other = other.compound
        return self.compound == other
