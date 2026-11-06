
karel_obv_token_api_dict = {
    'frontIsClear': 'front_is_clear',
    'leftIsClear': 'left_is_clear',
    'rightIsClear': 'right_is_clear',
    'markersPresent': 'marker_present',
    'noMarkersPresent': 'no_marker_present'
}

karel_action_token_list = ['move', 'turnLeft', 'turnRight', 'pickMarker', 'putMarker']

list_obv_token_api_dict = {
    'cursorAtStart': 'cursor_at_start',
    'cursorAtEnd': 'cursor_at_end',
    'listEmpty': 'list_empty',
    'elemIsZero': 'elem_is_zero',
    'elemLtNext': 'elem_lt_next',
}

from list_env.list_world import LIST_ACTIONS
list_action_token_list = LIST_ACTIONS
# NOTE: keep the environment names same as config.env_name in config/arguments.py
envs = ['karel', 'list']
obv_token_api_dict = {'karel': karel_obv_token_api_dict, 'list': list_obv_token_api_dict}
action_token_list = {'karel': karel_action_token_list, 'list': list_action_token_list}


class DSLData(object):

    def __init__(self, environment='karel', version=1):
        # Stored so DSLBase.get_yacc can pick a per-env tabmodule name
        # (prevents cross-job _parsetab.py thrash when multiple env generators
        # run concurrently against a shared filesystem).
        self.environment = environment
        self.version = version
        if version not in (1, 2) or (version == 2 and environment != 'karel'):
            raise ValueError(f"No DSL version {version} for environment '{environment}'")
        if environment == 'karel':
            self.tokens = [
                'DEF', 'RUN', 'M_LBRACE', 'M_RBRACE',
                'MOVE', 'TURN_RIGHT', 'TURN_LEFT',
                'PICK_MARKER', 'PUT_MARKER',
                'R_LBRACE', 'R_RBRACE',
                'INT',  # 'NEWLINE', 'SEMI',
                'REPEAT',
                'C_LBRACE', 'C_RBRACE',
                'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
                'IF', 'IFELSE', 'ELSE',
                'FRONT_IS_CLEAR', 'LEFT_IS_CLEAR', 'RIGHT_IS_CLEAR',
                'MARKERS_PRESENT', 'NO_MARKERS_PRESENT',
                'NOT',
                'W_LBRACE', 'W_RBRACE',
                'WHILE',
            ]

            self.t_FRONT_IS_CLEAR = 'frontIsClear'
            self.t_LEFT_IS_CLEAR = 'leftIsClear'
            self.t_RIGHT_IS_CLEAR = 'rightIsClear'
            self.t_MARKERS_PRESENT = 'markersPresent'
            self.t_NO_MARKERS_PRESENT = 'noMarkersPresent'

            self.conditional_functions = [
                self.t_FRONT_IS_CLEAR, self.t_LEFT_IS_CLEAR, self.t_RIGHT_IS_CLEAR,
                self.t_MARKERS_PRESENT, self.t_NO_MARKERS_PRESENT,
            ]

            self.conditional_functions_dict = {
                self.t_FRONT_IS_CLEAR: lambda x: x.front_is_clear, self.t_LEFT_IS_CLEAR: lambda x: x.left_is_clear,
                self.t_RIGHT_IS_CLEAR: lambda x: x.right_is_clear, self.t_MARKERS_PRESENT: lambda x: x.marker_present,
                self.t_NO_MARKERS_PRESENT: lambda x: x.no_marker_present,
            }

            self.t_MOVE = 'move'
            self.t_TURN_RIGHT = 'turnRight'
            self.t_TURN_LEFT = 'turnLeft'
            self.t_PICK_MARKER = 'pickMarker'
            self.t_PUT_MARKER = 'putMarker'

            self.action_functions = [
                self.t_MOVE,
                self.t_TURN_LEFT, self.t_TURN_RIGHT,
                self.t_PICK_MARKER, self.t_PUT_MARKER,
            ]

            # This dictionary is used for YACC parser in dsl_base.py for grammar
            # It seems that yacc parser uses these docstrings to learn the perception and action
            # tokens. Check karel_env/dsl/third-party/yacc.py:3139
            self.dynamic_docstring = {
                'cond_without_not': '''cond_without_not : FRONT_IS_CLEAR
                | LEFT_IS_CLEAR
                | RIGHT_IS_CLEAR
                | MARKERS_PRESENT
                | NO_MARKERS_PRESENT
                ''',
                'action': '''action : MOVE
                    | TURN_RIGHT
                    | TURN_LEFT
                    | PICK_MARKER
                    | PUT_MARKER
                    '''
            }
            if version == 2:
                self._drop_no_markers_present()
        elif environment == 'list':
            acts = [('CURSOR_LEFT', 'cursorLeft'), ('CURSOR_RIGHT', 'cursorRight'), ('PUSH_ZERO', 'pushZero'),
                    ('POP', 'pop'), ('SWAP_NEXT', 'swapNext'), ('INC_ELEM', 'incElem'), ('DEC_ELEM', 'decElem')]
            # (name, token, sampling weight): skewed toward the more discriminative conditions;
            # cursorAtStart is trivially true at the start of execution, so it is biased against.
            conds = [('CURSOR_AT_START', 'cursorAtStart', 0.10), ('CURSOR_AT_END', 'cursorAtEnd', 0.20),
                     ('LIST_EMPTY', 'listEmpty', 0.15), ('ELEM_IS_ZERO', 'elemIsZero', 0.20),
                     ('ELEM_LT_NEXT', 'elemLtNext', 0.35)]
            self.tokens = [
                'DEF', 'RUN', 'M_LBRACE', 'M_RBRACE',
                *[name for name, _ in acts],
                'R_LBRACE', 'R_RBRACE',
                'INT',
                'REPEAT',
                'C_LBRACE', 'C_RBRACE',
                'I_LBRACE', 'I_RBRACE', 'E_LBRACE', 'E_RBRACE',
                'IF', 'IFELSE', 'ELSE',
                *[name for name, _, _ in conds],
                'NOT',
                'W_LBRACE', 'W_RBRACE',
                'WHILE',
            ]
            for name, tok in acts + [c[:2] for c in conds]:
                setattr(self, 't_' + name, tok)
            self.conditional_functions = [tok for _, tok, _ in conds]
            self.conditional_functions_dict = {tok: (lambda x, api=list_obv_token_api_dict[tok]: getattr(x, api))
                                               for _, tok, _ in conds}
            self.action_functions = [tok for _, tok in acts]
            self.dynamic_docstring = {
                'cond_without_not': 'cond_without_not : ' + '\n                | '.join(n for n, _, _ in conds) + '\n                ',
                'action': 'action : ' + '\n                    | '.join(n for n, _ in acts) + '\n                    ',
            }
            # Override the class-level prob_action (length 5, set in DSLProb): one entry per action
            # (np.random.choice in random_tokens). Condition weights are renormalised when some are dropped.
            self.prob_action = [1.0 / len(acts)] * len(acts)
            weights = [w for _, _, w in conds]
            self.prob_cond_without_not = weights if len(conds) == 5 else [w / sum(weights) for w in weights]

        else:
            raise ValueError(f"Unknown environment '{environment}'")


    def _drop_no_markers_present(self):
        """Karel DSL v2: remove `noMarkersPresent`, which is exactly `not markersPresent`.

        Token ids after it shift by one (vocab 50 -> 49), so v2 datasets and
        checkpoints are not interchangeable with v1 ones. The dataset records its
        version in data/<datadir>/dsl_info.json (see karel_env.dsl.dsl_version_for).
        `markersPresent` absorbs the removed sampling mass so marker conditions keep
        their overall share (0.2) of sampled conditions.
        """
        self.tokens.remove('NO_MARKERS_PRESENT')
        del self.t_NO_MARKERS_PRESENT
        self.conditional_functions.remove('noMarkersPresent')
        del self.conditional_functions_dict['noMarkersPresent']
        self.dynamic_docstring['cond_without_not'] = '''cond_without_not : FRONT_IS_CLEAR
                | LEFT_IS_CLEAR
                | RIGHT_IS_CLEAR
                | MARKERS_PRESENT
                '''
        self.prob_cond_without_not = [0.5, 0.15, 0.15, 0.2]
