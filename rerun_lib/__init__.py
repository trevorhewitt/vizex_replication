"""Scripted, non-interactive rerun of every Chapter 5 analysis (see RERUN.md).

The modules port the logic of the notebooks in 011_notebooks_v2/ so that the
whole chapter can be regenerated from the committed CSVs with one command:

    python rerun.py

The notebooks are kept unchanged apart from an in_analysis guard; the scripts
are the reference implementation for rerun_outputs/.
"""
