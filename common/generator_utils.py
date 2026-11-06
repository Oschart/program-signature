"""Helpers shared by the dataset generators (karel_env/generator_mp.py, list_env/generator_mp.py).

Demo coverage: parse_and_trace's record_dict holds, per IF/IFELSE/WHILE node, which
outcomes the run(s) since the last reset took. A generator resets it before every demo
attempt and keeps a set of demos only if they TOGETHER take both outcomes of every node.

Split: a fixed, seeded train/test split written next to data.hdf5 (train_ids.txt /
test_ids.txt) plus dataset_info.json, which the loader (karel_env/program_dataset.py)
reads for the split and the DSL version.
"""
import json
import os

import numpy as np


def branch_sides(record_dict, only_covered=False):
    """(node position, outcome) pairs of every IF/IFELSE/WHILE node in the program;
    with only_covered, just the outcomes the last run(s) actually took."""
    return frozenset((v[0][0], side) for v in record_dict.values() for side in (True, False)
                     if not only_covered or v[0][1][side])


def reset_coverage(record_dict):
    for v in record_dict.values():
        v[0][1][True] = v[0][1][False] = False


def choose_covering_demos(candidates, k, all_sides):
    """Pick k candidate demos (each (s_h, a_h, covered_sides)) whose covered sides
    together include every branch outcome; None if the candidates can't (greedy cover).
    Returned in trial order."""
    if len(candidates) < k:
        return None
    chosen, missing = [], set(all_sides)
    while missing:
        gain, best = max((len(c[2] & missing), i) for i, c in enumerate(candidates) if i not in chosen)
        if gain == 0 or len(chosen) == k:
            return None
        chosen.append(best)
        missing -= candidates[best][2]
    chosen += [i for i in range(len(candidates)) if i not in chosen][:k - len(chosen)]
    return [candidates[i] for i in sorted(chosen)]


def write_split_and_info(config, ids, dsl, environment):
    """Fixed, seeded train/test split (train_ids.txt / test_ids.txt; there is no separate
    validation split) and dataset_info.json, which the loader reads for the split and the
    DSL version (karel_env/program_dataset.py). Train ids are in random order, so
    --size_cap prefixes are random subsets. Environments without DSL versions record 1."""
    perm = np.random.RandomState(config.seed).permutation(len(ids))
    splits = {'test': [ids[i] for i in perm[:config.num_test]],
              'train': [ids[i] for i in perm[config.num_test:]]}
    for name, split_ids in splits.items():
        with open(os.path.join(config.dir_name, f'{name}_ids.txt'), 'w') as f:
            f.write('\n'.join(split_ids) + '\n')
    info = {
        'environment': environment,
        'dsl_version': getattr(config, 'dsl_version', 1),
        'int2token': list(dsl.int2token),
        'split': {'train': 'train_ids.txt', 'test': 'test_ids.txt', 'seed': config.seed,
                  'n_train': len(splits['train']), 'n_test': len(splits['test'])},
        'generator_args': vars(config),
    }
    with open(os.path.join(config.dir_name, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=1)
