"""Decoders for the independent label-efficient segmentation pipeline."""

from adasam.models.decoder.lightweight_decoder import LightweightSemanticDecoder
from adasam.models.decoder.boundary_aware_decoder import BoundaryAwareSemanticDecoder

__all__ = ["LightweightSemanticDecoder", "BoundaryAwareSemanticDecoder"]
