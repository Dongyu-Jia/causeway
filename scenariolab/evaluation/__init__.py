"""Evaluation of the quarterly stack (SIM_DESIGN §3.9, §8, §10.1).

- ``harness``  the oracle-macro evaluation harness (port of the research round's simharness.py): conditional paths
  under realized factor paths, walk-forward metrics with block-bootstrap intervals, the placebo-corrected valuation
  anatomy, and adapters for the quarterly stack's models (the asset layer's L0).

- ``narrative``  the permanent end-to-end narrative test (SIM_DESIGN §8; P7): the episode registry with the sealed
  key, the blind prompt builder, estimates and confirmed runs, scoring on the quarterly stack and the monthly engine,
  the blindness audit, the report the evaluation tab renders, and the research round's imported run.

The pre-registered protocols (protocol.py) and the prospective ledger (ledger.py) come with the later phases. Nothing
here calls an LLM except a narrative-test run the user confirmed after its estimate (tests use the mock backend).
"""
