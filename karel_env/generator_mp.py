from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# import os
# os.environ["LINE_PROFILE"] = "1"
import h5py
import os
import argparse
from tqdm import tqdm
import random
import pickle
import torch as th
import numpy as np
import shutil
import multiprocessing
import math

# Project imports
from karel_env.dsl import get_DSL
from karel_env.dsl.dsl_parse_and_trace import parse_and_trace
from karel_env.dsl.canonicalize import canonicalize
from common.generator_utils import branch_sides, reset_coverage, choose_covering_demos, write_split_and_info
from karel_env.util import log
import karel_env.karel as karel
from time import time
from karel_env.constants import arr_desc
from common.utils import pad_and_stack_numpy
# from line_profiler import profile


class KarelStateGenerator(object):
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def print_state(self, state=None):
        agent_direction = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
        state_2d = np.chararray(state.shape[:2])
        state_2d[:] = '.'
        state_2d[state[:, :, 4]] = 'x'
        state_2d[state[:, :, 6]] = 'M'
        x, y, z = np.where(state[:, :, :4] > 0)
        state_2d[x[0], y[0]] = agent_direction[z[0]]

        state_2d = state_2d.decode()
        for i in range(state_2d.shape[0]):
            print("".join(state_2d[i]))

    # generate an initial env
    def generate_single_state(self, h=8, w=8, wall_prob=0.1, env_task_metadata={}):
        s = np.zeros([h, w, 16]) > 0
        # Wall
        s[:, :, 4] = self.rng.rand(h, w) > 1 - wall_prob
        s[0, :, 4] = True
        s[h - 1, :, 4] = True
        s[:, 0, 4] = True
        s[:, w - 1, 4] = True
        # Karel initial location
        valid_loc = False
        while (not valid_loc):
            y = self.rng.randint(0, h)
            x = self.rng.randint(0, w)
            if not s[y, x, 4]:
                valid_loc = True
                s[y, x, self.rng.randint(0, 4)] = True
        # Marker: num of max marker == 1 for now
        s[:, :, 6] = (self.rng.rand(h, w) > 0.9) * (s[:, :, 4] == False) > 0
        s[:, :, 5] = 1 - (np.sum(s[:, :, 6:], axis=-1) > 0) > 0
        assert np.sum(s[:, :, 5:]) == h * w, np.sum(s[:, :, :5])
        marker_weight = np.reshape(np.array(range(11)), (1, 1, 11))
        return s, y, x, np.sum(s[:, :, 4]), np.sum(marker_weight * s[:, :, 5:])

    # ... [Other Generator Methods kept identical for brevity] ...
    # (Note: In a real deployment, keep all methods provided in the original file. 
    #  For this solution, I assume the generator methods like `generate_single_state_clean_house` 
    #  remain defined exactly as in your upload inside this class.)


# @profile
def generate_worker_chunk(worker_id, num_samples_target, config, seed):
    """
    Worker function to generate a chunk of data.
    Writes to a temporary HDF5 file.
    """
    # Setup unique temporary filename for this worker
    temp_filename = os.path.join(config.dir_name, 'temp_worker_{}.hdf5'.format(worker_id))
    
    # Initialize tools with worker-specific seed
    dsl = get_DSL(dsl_type='prob', seed=seed, environment='karel', version=config.dsl_version)
    s_gen = KarelStateGenerator(seed=seed)
    karel_world = karel.Karel_world()

    # p_v_size = len(karel_world.get_perception_list())

    # Open temp file
    f = h5py.File(temp_filename, 'w')
    
    # Local tracking
    count = 0
    seen_programs = set()
    
    # Metrics
    max_demo_length_in_dataset = -1
    max_program_length_in_dataset = -1
    min_demo_length_in_dataset = float('inf')
    min_program_length_in_dataset = float('inf')

    # pbar = tqdm(total=num_samples_target, position=worker_id, desc=f"Worker {worker_id}",  mininterval=30.0, maxinterval=60.0)
    start_time = time()
    # Generation Loop
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

        # Local Deduplication
        if random_code in seen_programs:
            continue

        program_seq = np.array(dsl.code2intseq(random_code), dtype=np.int8)
        if program_seq.shape[0] > config.max_program_len:
            continue

        # Trace
        record_dict = {}
        if config.cover_all_branches_in_demos:
            exe, s_exe, record_dict = parse_and_trace(random_code, environment='karel')
            if len(record_dict) == 0 and ('WHILE' in random_code or 'IF' in random_code):
                 # Skip invalid parsings gracefully
                continue
            if not s_exe:
                continue

        # Demos: every attempt gets a fresh coverage record; keep the ones of valid
        # length and stop once k of them TOGETHER take both outcomes of every branch.
        all_sides = branch_sides(record_dict) if config.cover_all_branches_in_demos else frozenset()
        candidates, chosen, num_trial = [], None, 0
        while chosen is None and num_trial < config.max_demo_generation_trial:
            num_trial += 1
            try:
                s, _, _, _, _ = s_gen.generate_single_state(config.height, config.width, config.wall_prob)
                karel_world.set_new_state(s)
                if not config.cover_all_branches_in_demos:
                    dsl.run(karel_world, random_code)
                else:
                    reset_coverage(record_dict)
                    karel_world.clear_history()
                    karel_world, n, s_run = exe(karel_world, 0, record_dict, exe)
                    if not s_run: raise RuntimeError
            except RuntimeError:
                continue
            if not config.min_demo_length <= len(karel_world.s_h) <= config.max_demo_length:
                continue
            covered = branch_sides(record_dict, only_covered=True) if config.cover_all_branches_in_demos else frozenset()
            candidates.append(([arr_desc(st) for st in karel_world.s_h], np.array(karel_world.a_h), covered))
            chosen = choose_covering_demos(candidates, config.num_demo_per_program, all_sides)

        if chosen is None:
            continue
        s_h_list = [c[0] for c in chosen]
        a_h_list = [c[1] for c in chosen]
        num_demo = len(chosen)

        # Data Packaging
        len_s_h = np.array([len(s_h) for s_h in s_h_list], dtype=np.int16)
        if np.max(len_s_h) < config.min_max_demo_length_for_program:
            continue

        demos_s_h = pad_and_stack_numpy(s_h_list)
        # demos_s_h = s_h_list
        # demos_s_h = np.zeros([num_demo, np.max(len_s_h), config.height, config.width, len(karel.state_table)], dtype=bool)
        # demos_p_v_h = np.zeros([num_demo, np.max(len_s_h), p_v_size], dtype=bool)
        # for i, s_h_len in enumerate(len_s_h):
        #     # demos_s_h[i, :s_h.shape[0]] = s_h
        #     demos_p_v_h[i, :s_h_len] = p_v_list[i]

        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list], dtype=np.int16)
        demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.int8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h
        
        # Save to temp file
        # ID is just local index, will be renamed in merge
        grp_name = 'local_{}'.format(count)
        grp = f.create_group(grp_name)
        grp['program'] = program_seq
        grp['s_h_len'] = len_s_h
        grp['a_h_len'] = len_a_h
        grp['s_h'] = demos_s_h
        # grp['p_v_h'] = demos_p_v_h
        grp['a_h'] = demos_a_h
        # Save the string code to attributes for easy duplicate checking during merge
        grp.attrs['code_str'] = random_code
        
        # Update local stats
        max_demo_length_in_dataset = max(max_demo_length_in_dataset, np.max(len_s_h))
        max_program_length_in_dataset = max(max_program_length_in_dataset, program_seq.shape[0])
        min_demo_length_in_dataset = min(min_demo_length_in_dataset, np.min(len_s_h))
        min_program_length_in_dataset = min(min_program_length_in_dataset, program_seq.shape[0])

        seen_programs.add(random_code)
        count += 1

        # --- UPDATE PROGRESS BAR ---
        # pbar.update(1)
        elapsed = time() - start_time
        time_per_sample = elapsed / count
        est_total_time = time_per_sample * num_samples_target
        est_remaining = (est_total_time - elapsed)/3600.0
        print(f"Worker [{worker_id}] generated {count}/{num_samples_target} samples. Per-sample={time_per_sample:.2f}s. Elapsed={elapsed:.2f}s, Remaining={est_remaining:.2f} hrs")
    
    # Save worker metadata
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
    # Build the parser cache in the main process: a new grammar (e.g. DSL v2) would
    # otherwise have all workers racing to write the same _parsetab file.
    get_DSL(dsl_type='prob', seed=config.seed, environment='karel', version=config.dsl_version)

    # Files for final output
    f_main = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'w')
    id_file = open(os.path.join(dir_name, 'id.txt'), 'w')
    
    # Global tracking
    global_seen_programs = set()
    global_count = 0
    all_ids = []
    current_base_seed = config.seed
    
    # Global metadata trackers
    g_max_demo_len = -1
    g_min_demo_len = float('inf')
    g_max_prog_len = -1
    g_min_prog_len = float('inf')
    
    print(f"Goal: {num_total} unique programs across {num_workers} workers.")

    # --- OUTER LOOP: KEEPS RELAUNCHING UNTIL TARGET MET ---
    iteration = 1
    while global_count < num_total:
        needed = num_total - global_count
        # Oversample by 20% to account for duplicates we are about to filter
        target_per_round = int(needed * 1.2) 
        # Ensure at least 1 per worker if we are very close
        target_per_round = max(target_per_round, num_workers)
        
        chunk_size = math.ceil(target_per_round / num_workers)
        
        print(f"\n--- Round {iteration} ---")
        print(f"Current count: {global_count}/{num_total}")
        print(f"Generating approx {target_per_round} new candidates (Chunk size: {chunk_size})")
        print(f"Using base seed: {current_base_seed}")
        
        # Prepare tasks with NEW SEEDS
        pool = multiprocessing.Pool(processes=num_workers)
        tasks = []
        for i in range(num_workers):
            # Crucial: Shift the seed for every worker and every round
            worker_seed = current_base_seed + i + (iteration * 1000) 
            tasks.append((i, chunk_size, config, worker_seed))

        # Launch workers
        temp_files = pool.starmap(generate_worker_chunk, tasks)
        pool.close()
        pool.join()
        
        print(f"\nRound {iteration} finished. Merging into main dataset...")

        # Merge Phase
        pbar = tqdm(total=len(temp_files), desc="Processing Temp Files")
        
        for t_file in temp_files:
            if not os.path.exists(t_file):
                continue
                
            try:
                f_temp = h5py.File(t_file, 'r')
                keys = [k for k in f_temp.keys() if k != 'worker_info']
                
                # We iterate through the temp file and take only what we need
                for k in keys:
                    if global_count >= num_total:
                        break # Stop if we filled the quota
                    
                    grp_src = f_temp[k]
                    code_str = grp_src.attrs.get('code_str')
                    
                    if code_str is None or code_str in global_seen_programs:
                        continue # Duplicate detected!
                    
                    # New unique program found
                    global_seen_programs.add(code_str)
                    
                    # Copy to main file
                    program_seq = grp_src['program'][()]
                    s_h_len = grp_src['s_h_len'][()]
                    
                    new_id = 'no_{}_prog_len_{}_max_s_h_len_{}'.format(
                        global_count, program_seq.shape[0], np.max(s_h_len))
                    
                    id_file.write(new_id + '\n')
                    all_ids.append(new_id)
                    grp_dst = f_main.create_group(new_id)
                    
                    for dset in ['program', 's_h_len', 'a_h_len', 's_h', 'a_h']:
                        grp_dst.create_dataset(dset, data=grp_src[dset][()])
                    
                    # Update Stats
                    g_max_demo_len = max(g_max_demo_len, np.max(grp_src['s_h_len']))
                    g_min_demo_len = min(g_min_demo_len, np.min(grp_src['s_h_len']))
                    g_max_prog_len = max(g_max_prog_len, program_seq.shape[0])
                    g_min_prog_len = min(g_min_prog_len, program_seq.shape[0])

                    global_count += 1

                f_temp.close()
                os.remove(t_file) # Cleanup immediately
                
            except Exception as e:
                print(f"Error reading temp file {t_file}: {e}")
            
            pbar.update(1)
        
        pbar.close()
        
        # Prepare for potential next round
        current_base_seed += num_workers # Ensure next batch has totally different seeds
        iteration += 1

    # Final Metadata Writing
    dsl = get_DSL(dsl_type='prob', seed=config.seed, environment='karel', version=config.dsl_version)

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
    grp['dsl_version'] = config.dsl_version

    f_main.close()
    id_file.close()
    write_split_and_info(config, all_ids, dsl, 'karel')
    
    log.info('Dataset generated under {} with {} unique samples'.format(dir_name, global_count))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='karel_dataset')
    parser.add_argument('--height', type=int, default=8,
                        help='height of square grid world')
    parser.add_argument('--width', type=int, default=8,
                        help='width of square grid world')
    parser.add_argument('--num_train', type=int, default=25000, help='num train')
    parser.add_argument('--num_test', '--num_val', dest='num_test', type=int, default=5000,
                        help='size of the held-out test split (there is no separate validation split)')
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel workers')
    parser.add_argument('--wall_prob', type=float, default=0.1,
                        help='probability of wall generation')
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
    parser.add_argument('--cover_all_branches_in_demos', action='store_true',
                        help='the stored demos must together take both outcomes of every IF/IFELSE/WHILE '
                             '(and programs may not contain REPEAT R=0), so every statement runs in some demo')
    parser.add_argument('--canonicalize', action='store_true',
                        help='rewrite sampled programs to a trace-equivalent canonical form before tracing '
                             '(karel_env/dsl/canonicalize.py); dedup is on the canonical string')
    parser.add_argument('--dsl_version', type=int, default=1, choices=[1, 2],
                        help='Karel DSL version; 2 drops noMarkersPresent (== not markersPresent), vocab 50 -> 49')
    parser.add_argument('--just_testing', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    args.dir_name = os.path.join('data/', args.dir_name)
    os.makedirs(args.dir_name, exist_ok=True)

    generator_parallel(args)

if __name__ == '__main__':
    main()