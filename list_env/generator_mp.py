from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import argparse
from tqdm import tqdm
import numpy as np
import multiprocessing
import math

from karel_env.dsl import get_DSL
from karel_env.dsl.dsl_parse_and_trace import parse_and_trace
from karel_env.dsl.canonicalize import canonicalize
from karel_env.util import log
from list_env.list_world import ListWorld
from list_env.state_generator import ListStateGenerator
from common.generator_utils import branch_sides, reset_coverage, choose_covering_demos, write_split_and_info
from time import time


# Programs need at least one of these (cursor-only walks are trivial); per environment.
STRUCTURAL_TOKENS = {'list': ('pushZero', 'pop', 'swapNext')}
WORLDS = {'list': ListWorld}


def required_tokens(config):
    """A program needs one of these."""
    return STRUCTURAL_TOKENS[config.environment]


def generate_worker_chunk(worker_id, num_samples_target, config, seed):
    """
    Worker function to generate a chunk of list-manipulation data.
    Writes to a temporary HDF5 file.
    """
    temp_filename = os.path.join(config.dir_name, 'temp_worker_{}.hdf5'.format(worker_id))

    dsl = get_DSL(dsl_type='prob', seed=seed, environment=config.environment)
    s_gen = ListStateGenerator(seed=seed, L_max=config.L_max, max_val=config.max_val,
                               min_init_len=config.min_init_len)
    world = WORLDS[config.environment](L_max=config.L_max, max_val=config.max_val)

    f = h5py.File(temp_filename, 'w')

    count = 0
    seen_programs = set()

    max_demo_length_in_dataset = -1
    max_program_length_in_dataset = -1
    min_demo_length_in_dataset = float('inf')
    min_program_length_in_dataset = float('inf')

    start_time = time()

    while count < num_samples_target:
        random_code = dsl.random_code(max_depth=config.max_program_stmt_depth,
                                      max_nesting_depth=config.max_program_nesting_depth)
        if config.canonicalize:
            random_code = canonicalize(random_code)  # trace-equivalent; None = no valid form
            if random_code is None:
                continue
        # REPEAT R=0 bodies never run, so no demo set could cover them
        if config.cover_all_branches_in_demos and 'R=0' in random_code.split():
            continue

        if random_code in seen_programs:
            continue

        # Filter out programs with no structural op (cursor-only walks are trivial).
        if not any(tok in random_code for tok in required_tokens(config)):
            continue

        program_seq = np.array(dsl.code2intseq(random_code), dtype=np.int8)
        if program_seq.shape[0] > config.max_program_len:
            continue

        # Branch tracing
        record_dict = {}
        if config.cover_all_branches_in_demos:
            exe, s_exe, record_dict = parse_and_trace(random_code, environment=config.environment)
            if len(record_dict) == 0 and ('WHILE' in random_code or 'IF' in random_code):
                continue
            if not s_exe:
                continue

        # Demos: every attempt gets a fresh coverage record; keep the ones of valid length
        # and state uniqueness, and stop once k of them TOGETHER take both outcomes of
        # every branch.
        all_sides = branch_sides(record_dict) if config.cover_all_branches_in_demos else frozenset()
        candidates, chosen, num_trial = [], None, 0
        while chosen is None and num_trial < config.max_demo_generation_trial:
            num_trial += 1
            try:
                values, cursor = s_gen.generate_single_state()
                world.set_new_state(values, cursor)
                if not config.cover_all_branches_in_demos:
                    dsl.run(world, random_code)
                else:
                    reset_coverage(record_dict)
                    world.clear_history()
                    world, n, s_run = exe(world, 0, record_dict, exe)
                    if not s_run:
                        raise RuntimeError
            except RuntimeError:
                continue
            if not config.min_demo_length <= len(world.s_h) <= config.max_demo_length:
                continue
            # 1.0 also rules out no-op steps, so the states alone determine every action
            if config.min_state_uniqueness > 0 and \
                    len(set(map(bytes, world.s_h))) / len(world.s_h) < config.min_state_uniqueness:
                continue
            covered = branch_sides(record_dict, only_covered=True) if config.cover_all_branches_in_demos else frozenset()
            candidates.append((np.stack(world.s_h, axis=0), np.array(world.a_h, dtype=np.int8), covered))
            chosen = choose_covering_demos(candidates, config.num_demo_per_program, all_sides)

        if chosen is None:
            continue
        s_h_list = [c[0] for c in chosen]  # (T, L_max+2) each
        a_h_list = [c[1] for c in chosen]
        num_demo = len(chosen)

        # Data packaging
        len_s_h = np.array([s_h.shape[0] for s_h in s_h_list], dtype=np.int16)
        if np.max(len_s_h) < config.min_max_demo_length_for_program:
            continue

        max_t = np.max(len_s_h)
        state_dim = config.L_max + 2
        demos_s_h = np.zeros([num_demo, max_t, state_dim], dtype=np.int8)
        for i, s_h in enumerate(s_h_list):
            demos_s_h[i, :s_h.shape[0]] = s_h

        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)
        demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.int8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h

        # Save to temp file
        grp_name = 'local_{}'.format(count)
        grp = f.create_group(grp_name)
        grp['program'] = program_seq
        grp['s_h_len'] = len_s_h
        grp['a_h_len'] = len_a_h
        grp['s_h'] = demos_s_h
        grp['a_h'] = demos_a_h
        grp.attrs['code_str'] = random_code

        max_demo_length_in_dataset = max(max_demo_length_in_dataset, np.max(len_s_h))
        max_program_length_in_dataset = max(max_program_length_in_dataset, program_seq.shape[0])
        min_demo_length_in_dataset = min(min_demo_length_in_dataset, np.min(len_s_h))
        min_program_length_in_dataset = min(min_program_length_in_dataset, program_seq.shape[0])

        seen_programs.add(random_code)
        count += 1

        elapsed = time() - start_time
        time_per_sample = elapsed / count
        est_total_time = time_per_sample * num_samples_target
        est_remaining = (est_total_time - elapsed) / 3600.0
        print(f"Worker [{worker_id}] generated {count}/{num_samples_target} samples. "
              f"Per-sample={time_per_sample:.2f}s. Elapsed={elapsed:.2f}s, Remaining={est_remaining:.2f} hrs")

    grp = f.create_group('worker_info')
    grp['max_demo_length'] = max_demo_length_in_dataset
    grp['min_demo_length'] = min_demo_length_in_dataset
    grp['max_program_len'] = max_program_length_in_dataset
    grp['min_program_length'] = min_program_length_in_dataset

    f.close()
    return temp_filename


def generator_parallel(config):
    dir_name = config.dir_name
    num_train = config.num_train
    num_test = config.num_test
    num_total = num_train + num_test
    num_workers = config.num_workers
    # Build the parser cache in the main process so workers don't race to write it.
    get_DSL(dsl_type='prob', seed=config.seed, environment=config.environment)

    f_main = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'w')
    id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

    global_seen_programs = set()
    global_count = 0
    all_ids = []
    current_base_seed = config.seed

    g_max_demo_len = -1
    g_min_demo_len = float('inf')
    g_max_prog_len = -1
    g_min_prog_len = float('inf')

    print(f"Goal: {num_total} unique programs across {num_workers} workers.")

    iteration = 1
    while global_count < num_total:
        needed = num_total - global_count
        target_per_round = int(needed * 1.2)
        target_per_round = max(target_per_round, num_workers)

        chunk_size = math.ceil(target_per_round / num_workers)

        print(f"\n--- Round {iteration} ---")
        print(f"Current count: {global_count}/{num_total}")
        print(f"Generating approx {target_per_round} new candidates (Chunk size: {chunk_size})")
        print(f"Using base seed: {current_base_seed}")

        pool = multiprocessing.Pool(processes=num_workers)
        tasks = []
        for i in range(num_workers):
            worker_seed = current_base_seed + i + (iteration * 1000)
            tasks.append((i, chunk_size, config, worker_seed))

        temp_files = pool.starmap(generate_worker_chunk, tasks)
        pool.close()
        pool.join()

        print(f"\nRound {iteration} finished. Merging into main dataset...")

        pbar = tqdm(total=len(temp_files), desc="Processing Temp Files")

        for t_file in temp_files:
            if not os.path.exists(t_file):
                continue

            try:
                f_temp = h5py.File(t_file, 'r')
                keys = [k for k in f_temp.keys() if k != 'worker_info']

                for k in keys:
                    if global_count >= num_total:
                        break

                    grp_src = f_temp[k]
                    code_str = grp_src.attrs.get('code_str')

                    if code_str is None or code_str in global_seen_programs:
                        continue

                    global_seen_programs.add(code_str)

                    program_seq = grp_src['program'][()]
                    s_h_len = grp_src['s_h_len'][()]

                    new_id = 'no_{}_prog_len_{}_max_s_h_len_{}'.format(
                        global_count, program_seq.shape[0], np.max(s_h_len))

                    id_file.write(new_id + '\n')
                    all_ids.append(new_id)
                    grp_dst = f_main.create_group(new_id)

                    for dset in ['program', 's_h_len', 'a_h_len', 's_h', 'a_h']:
                        grp_dst.create_dataset(dset, data=grp_src[dset][()])

                    g_max_demo_len = max(g_max_demo_len, np.max(grp_src['s_h_len']))
                    g_min_demo_len = min(g_min_demo_len, np.min(grp_src['s_h_len']))
                    g_max_prog_len = max(g_max_prog_len, program_seq.shape[0])
                    g_min_prog_len = min(g_min_prog_len, program_seq.shape[0])

                    global_count += 1

                f_temp.close()
                os.remove(t_file)

            except Exception as e:
                print(f"Error reading temp file {t_file}: {e}")

            pbar.update(1)

        pbar.close()

        current_base_seed += num_workers
        iteration += 1

    # Final metadata
    dsl = get_DSL(dsl_type='prob', seed=config.seed, environment=config.environment)

    grp = f_main.create_group('data_info')
    grp['max_demo_length'] = g_max_demo_len
    grp['min_demo_length'] = g_min_demo_len
    grp['dsl_type'] = 'prob'
    grp['max_program_len'] = g_max_prog_len
    grp['min_program_length'] = g_min_prog_len
    grp['num_program_tokens'] = len(dsl.int2token)
    grp['num_demo_per_program'] = config.num_demo_per_program
    grp['num_action_tokens'] = len(dsl.action_functions)
    grp['num_train'] = num_train
    grp['num_test'] = num_test
    grp['L_max'] = config.L_max
    grp['max_val'] = config.max_val

    f_main.close()
    id_file.close()
    write_split_and_info(config, all_ids, dsl, config.environment)

    log.info('Dataset generated under {} with {} unique samples'.format(dir_name, global_count))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='list_dataset')
    parser.add_argument('--environment', type=str, default='list', choices=['list'])
    parser.add_argument('--L_max', type=int, default=20,
                        help='maximum list length')
    parser.add_argument('--max_val', type=int, default=19,
                        help='maximum element value (inclusive)')
    parser.add_argument('--min_init_len', type=int, default=1,
                        help='minimum initial list length when sampling states')
    parser.add_argument('--num_train', type=int, default=25000, help='num train')
    parser.add_argument('--num_test', '--num_val', dest='num_test', type=int, default=5000,
                        help='size of the held-out test split (there is no separate validation split)')
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel workers')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--max_program_len', type=int, default=50)
    parser.add_argument('--max_program_stmt_depth', type=int, default=6)
    parser.add_argument('--max_program_nesting_depth', type=int, default=4)
    parser.add_argument('--min_max_demo_length_for_program', type=int, default=2)
    parser.add_argument('--min_demo_length', type=int, default=4,
                        help='min demo length')
    parser.add_argument('--max_demo_length', type=int, default=32,
                        help='max demo length')
    parser.add_argument('--num_demo_per_program', type=int, default=100,
                        help='number of seen demonstrations')
    parser.add_argument('--max_demo_generation_trial', type=int, default=200)
    parser.add_argument('--min_state_uniqueness', type=float, default=0.0,
                        help='minimum fraction of unique states in a demo (0.0 to disable, e.g. 0.8 for 80%%)')
    parser.add_argument('--cover_all_branches_in_demos', action='store_true',
                        help='the stored demos must together take both outcomes of every IF/IFELSE/WHILE '
                             '(and programs may not contain REPEAT R=0), so every statement runs in some demo')
    parser.add_argument('--canonicalize', action='store_true',
                        help='rewrite sampled programs to a trace-equivalent canonical form before tracing '
                             '(karel_env/dsl/canonicalize.py); dedup is on the canonical string')
    args = parser.parse_args()
    args.dir_name = os.path.join('data/', args.dir_name)
    os.makedirs(args.dir_name, exist_ok=True)

    generator_parallel(args)


if __name__ == '__main__':
    main()
