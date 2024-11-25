from diambra.arena import SpaceTypes, Roles

all_settings = {
    'basic': {
        'step_ratio': 3,
        'difficulty': 4,
        'frame_shape': (112, 192, 1),
        'role': Roles.P1,
        'characters': 'Ken',
        'action_space': SpaceTypes.MULTI_DISCRETE,
        'continue_game': 0.0
    },
    'wrapper': {
        'normalize_reward': True,
        'normalization_factor': 0.5,
        'stack_frames': 6,
        'stack_actions': 4,
        'scale': True,
        'exclude_image_scaling': True,
        'flatten': True,
        'filter_keys': ["own_health", "opp_health", "own_side", "opp_side", "opp_character", "stage", "timer"],
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