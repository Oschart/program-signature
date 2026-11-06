"""RASP environment.

Recover RASP programs from traces of their behaviour. Unlike the imperative envs
(karel/list), RASP is functional, so there is no simulated world: a trace is a
sweep (`sweep.py`), the program's outputs as one input token is raised through
every value, and a decoded program is correct if it reproduces every sweep.

Programs are evaluated natively (`rasp_grammar.evaluate_spec_py`), so neither
generation nor training needs Tracr.
"""
