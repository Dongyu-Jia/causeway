"""Evaluation of the quarterly stack (SIM_DESIGN §3.9, §8, §10.1).

- ``harness``  the oracle-macro evaluation harness (port of the research round's simharness.py): conditional paths
  under realized factor paths, walk-forward metrics with block-bootstrap intervals, the placebo-corrected valuation
  anatomy, and adapters for the quarterly stack's models (the asset layer's L0).

The pre-registered protocols (protocol.py), the narrative test (narrative/) and the prospective ledger (ledger.py)
come with the later phases. Nothing here calls an LLM.
"""
