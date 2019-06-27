import configparser

def get_arguments(filename='config'):
    config=configparser.ConfigParser()
    config.read(filenames=filename)

    arguments = {}
    arguments['is_fresh'] = config['DEFAULT'].getboolean('is_fresh')
    arguments['train_predictor'] = config['DEFAULT'].getboolean('train_predictor')
    arguments['predictor_steps'] = config['DEFAULT'].getint('predictor_steps')
    arguments['checkpoint_num'] = config['DEFAULT']['checkpoint_num']
    arguments['policy_checkpoint_num'] = config['DEFAULT']['policy_checkpoint_num']
    arguments['PPO_steps'] = config['DEFAULT'].getint('PPO_steps')
    arguments['predictor_dataset'] = config['DEFAULT'].getint('predictor_dataset')
    arguments['PPO_learning_rate'] = config['DEFAULT'].getfloat('PPO_learning_rate')
    arguments['predictor_type'] = config['DEFAULT']['predictor_type']
    arguments['reward_type'] = config['DEFAULT']['reward_type']
    # arguments['folder_name'] = config['DEFAULT']['folder_name']
    arguments['env_id'] = config['DEFAULT']['env_id']
    arguments['num_meta_iter'] = config['DEFAULT'].getint('num_meta_iter')
    arguments['only_test'] = config['DEFAULT'].getboolean('only_test')
    arguments['ball_type'] = config['DEFAULT'].getint('ball_type')
    arguments['start_state'] = config['DEFAULT']['start_state']
    arguments['flip_enabled'] = config['DEFAULT'].getboolean('flip_enabled')
    arguments['coverage_factor'] = config['DEFAULT'].getfloat('coverage_factor')
    arguments['reward_scale'] = config['DEFAULT'].getfloat('reward_scale')
    arguments['predictor_lr_steps'] = config['DEFAULT'].getint('predictor_lr_steps')
    arguments['chain_length'] = config['DEFAULT'].getint('chain_length')
    arguments['num_tries'] = config['DEFAULT'].getint('num_tries')
    arguments['predictor_loss'] = config['DEFAULT']['predictor_loss']
    arguments['enable_notification'] = config['DEFAULT'].getboolean('enable_notification')
    arguments['use_mass_distribution'] = config['DEFAULT'].getboolean('use_mass_distribution')
    arguments['mass_range_upper'] = config['DEFAULT'].getfloat('mass_range_upper')
    arguments['mass_range_lower'] = config['DEFAULT'].getfloat('mass_range_lower')
    return arguments
