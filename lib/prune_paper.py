"""Active paper entry point: strict structured pipeline with no Wanda scoring/masking."""
from .prune_paper_strict import PAPER_METHODS, prune_paper

__all__ = ["PAPER_METHODS", "prune_paper"]
