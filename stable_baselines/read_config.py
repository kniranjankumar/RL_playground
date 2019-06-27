import configparser

def get_arguments(filename='config', mode='DEFAULT'):
    config=configparser.ConfigParser()
    config.read(filenames=filename)

    arguments = {}
    arguments['is_fresh'] = config[mode].getboolean('is_fresh')
    arguments['train_predictor'] = config[mode].getboolean('train_predictor')
    arguments['predictor_steps'] = config[mode].getint('predictor_steps')
    arguments['checkpoint_num'] = config[mode]['checkpoint_num']
    arguments['policy_checkpoint_num'] = config[mode]['policy_checkpoint_num']
    arguments['PPO_steps'] = config[mode].getint('PPO_steps')
    arguments['predictor_dataset'] = config[mode].getint('predictor_dataset')
    arguments['PPO_learning_rate'] = config[mode].getfloat('PPO_learning_rate')
    arguments['predictor_type'] = config[mode]['predictor_type']
    arguments['reward_type'] = config[mode]['reward_type']
    # arguments['folder_name'] = config[mode]['folder_name']
    arguments['env_id'] = config[mode]['env_id']
    arguments['num_meta_iter'] = config[mode].getint('num_meta_iter')
    arguments['only_test'] = config[mode].getboolean('only_test')
    arguments['ball_type'] = config[mode].getint('ball_type')
    # arguments['start_state'] = config[mode]['start_state']
    arguments['start_state'] = None if config[mode]['start_state'] == 'None' else config[mode].getfloat('start_state')
    arguments['flip_enabled'] = config[mode].getboolean('flip_enabled')
    arguments['coverage_factor'] = config[mode].getfloat('coverage_factor')
    arguments['reward_scale'] = config[mode].getfloat('reward_scale')
    arguments['predictor_lr_steps'] = config[mode].getint('predictor_lr_steps')
    arguments['chain_length'] = config[mode].getint('chain_length')
    arguments['num_tries'] = config[mode].getint('num_tries')
    arguments['predictor_loss'] = config[mode]['predictor_loss']
    arguments['enable_notification'] = config[mode].getboolean('enable_notification')
    arguments['use_mass_distribution'] = config[mode].getboolean('use_mass_distribution')
    # arguments['mass_range_upper'] = config[mode].getfloat('mass_range_upper')
    # arguments['mass_range_lower'] = config[mode].getfloat('mass_range_lower')
    return arguments
