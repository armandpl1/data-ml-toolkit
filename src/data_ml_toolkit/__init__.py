"""Shared SDK for my data engineering and ML repositories.

Modules:
    data_ml_toolkit.logging  Structured JSON logging
    data_ml_toolkit.config   YAML config loading and repo-relative paths
    data_ml_toolkit.s3       Read and write S3 objects (install the `aws` extra)
    data_ml_toolkit.sync     Check that repos use the toolkit's tool versions
    data_ml_toolkit.llm      LLM calls with record/replay and a cost ledger (`llm` extra)
    data_ml_toolkit.evals    Golden sets, statistics, regressions, LLM judges (`evals` extra)
"""

from importlib.metadata import version

__version__ = version("data-ml-toolkit")
