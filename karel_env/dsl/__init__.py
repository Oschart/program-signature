import json
import os
import sys
sys.path.insert(0, 'karel_env/dsl')
from dsl_prob import DSLProb


def get_DSL(dsl_type='prob', seed=None, environment='karel', version=1):
    # RASP is a functional DSL with no yacc/DSLProb machinery — return the shim
    # regardless of dsl_type (BaseTrainer._make_meta calls with dsl_type='prob').
    if environment == 'rasp':
        from rasp_env.rasp_grammar import RaspDSL
        return RaspDSL(seed=seed)
    if dsl_type == 'prob':
        return DSLProb(seed=seed, environment=environment, version=version)
    else:
        raise ValueError('Undefined dsl type')


def dsl_version_for(datadir):
    """DSL version a dataset was generated with: data/<datadir>/dataset_info.json, else 1."""
    path = os.path.join('data', datadir, 'dataset_info.json')
    if not os.path.exists(path):
        return 1
    with open(path) as f:
        return int(json.load(f)['dsl_version'])
