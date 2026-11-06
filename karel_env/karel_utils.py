

from karel_env.dsl import get_DSL
from karel_env.dsl.dsl_prob import DSLProb


def execute_program_(config, program, s_0, a_gt=None):
    env_spec = config["env_spec"]
    world = env_spec.make_world(s_0, monitor_a=a_gt)
    dsl: DSLProb = config["dsl-env"]
    s_h = dsl.run_parsed(world, program)
    a_h = world.a_h
    return s_h, a_h

def verify_program(config, program_seq, s_gt, a_gt, a_gt_len):
    dsl = config["dsl-env"]
    eos_token = dsl.token2int['m)']
    # Get index of first eos token in tensor
    eos_idx = (program_seq == eos_token).nonzero(as_tuple=True)
    if len(eos_idx[0]) > 0:
        first_eos_pos = eos_idx[0][0].item()
        program_seq = program_seq[:first_eos_pos + 1]
    program = dsl.intseq2str(program_seq)
    if program[:3] != "DEF":
        program = f"DEF {program}"
    parsed_program = dsl.parse(program)
    K = s_gt.shape[0]
    match_count = 0
    for k in range(K):
        try:
            a_gt_k = a_gt[k][:a_gt_len[k]]
            s_h, a_h = execute_program_(config, parsed_program, s_gt[k,0], a_gt=a_gt_k)
            # The monitor catches mismatches only while both sequences are
            # emitting, so it misses two cases: (1) decoded program halts as a
            # strict prefix of GT; (2) decoded program emits extra actions past
            # len(monitor_a). Length equality + monitor-clean execution
            # together imply full sequence equality.
            if len(a_h) == int(a_gt_len[k]):
                match_count += 1
        except Exception as e:
            # Runtime error in program execution
            pass
    match_ratio = match_count / K
    return match_ratio

def verify_programs(config, program_seqs, s_gt, a_gt, a_gt_len):
    B, K = s_gt.shape[:2]
    total_match_cnt = 0
    match_ratio_sum = 0.0
    io_match_ratio_sum = 0.0  # TODO: actually implement the IO accr
    
    for i in range(B):
        match_ratio = verify_program(config, program_seqs[i], s_gt[i], a_gt[i], a_gt_len[i])
        total_match_cnt += (match_ratio == 1.0)
        match_ratio_sum += match_ratio
    ret_dict = {
        "total_match": total_match_cnt / B,
        "match_ratio": match_ratio_sum / B,
        "io_match_ratio": 0.0
    }
    return ret_dict