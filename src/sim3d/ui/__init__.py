"""Streamlit front end (spec sections 119, 121-124).

Streamlit is *only* a frontend.  No scientific logic lives here: every page
calls into :class:`sim3d.experiments.pipeline.Pipeline`, which is the same
code path the CLI uses.  Deleting this package would cost the project a
user interface and nothing else.
"""
