"""Reusable components for the standalone face-recognition evaluation apps.

The package provides the shared pipeline used by the CLI tools: dataset selection,
preprocessing and alignment, runtime execution, metric computation, and final reporting.
The common data flow is raw image -> detected face -> aligned tensor -> embedding vector
-> identity template -> verification metrics -> saved artefacts.
"""
