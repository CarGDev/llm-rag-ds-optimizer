"""Data source loaders for real corpora."""

from llmds.data_sources.msmarco import load_msmarco
from llmds.data_sources.beir_loader import load_beir
from llmds.data_sources.amazon_reviews import load_amazon_reviews
from llmds.data_sources.yelp import load_yelp
from llmds.data_sources.wikipedia import load_wikipedia
from llmds.data_sources.commoncrawl import load_commoncrawl

__all__ = [
    "load_msmarco",
    "load_beir",
    "load_amazon_reviews",
    "load_yelp",
    "load_wikipedia",
    "load_commoncrawl",
]

