from diambra.arena import SpaceTypes, Roles

all_settings = {
    'basic': {
        # 'step_ratio': 3,
        'step_ratio': 1,
        'difficulty': 8,
        'frame_shape': (112, 192, 1),
        'role': Roles.P1,
        'characters': 'Ken',
        'action_space': SpaceTypes.MULTI_DISCRETE,
        'continue_game': 0.0
    },
    'wrapper': {
        'normalize_reward': True,
        'normalization_factor': 0.5,
        # 'stack_frames': 9,
        # 'stack_actions': 6,
        'stack_frames': 24,
        'stack_actions': 12,
        # 'repeat_action': 8,
        'scale': True,
        'exclude_image_scaling': True,
        'flatten': True,
        'filter_keys': ["action",
                        "own_health", "opp_health",
                        "own_side", "opp_side",
                        "own_stun_bar", "opp_stun_bar",
                        "own_stunned", "opp_stunned",
                        "own_super_bar", "opp_super_bar",
                        "own_super_count", "opp_super_count",
                        "opp_character",
                        "stage", "timer"],
        'role_relative': True,
        'add_last_action': True
    },
    'agent': {
        'model_folder': './ckpts/',
        'log_dir': './logs/',
        'model_checkpoint': 'PPO',
        'autosave_freq': 31250,
        'time_steps': int(5e6)
    }
}