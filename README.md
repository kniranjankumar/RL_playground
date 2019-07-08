# RL_playground
This project is built upon [Stable baselines](https://github.com/hill-a/stable-baselines), [DartEnv](https://github.com/DartEnv/dart-env) and other libraries listed in requirements.

Requirements:
python 3.6
Dartsim 6.3
pydart2
pyquaternion
tensorflow 1.11
matplotlib 2.2.4
tqdm

Installation:
After installing all the dependencies, do the following
1. cd ./DartEnv2/
2. pip3 install -e '.[dart]'
3. cd ..
4. mkdir ./experiments
5. cd ./stable_baselines

To run_experiments with 2link chain mass range 3, run the following

python3 run_experiments.py --type 2link --folder_name 2link_train --mass_range 3

Use 3link instead to run 3link experiments and choose --mass_range 1 or 2 to try a different mass range.

The files relevant to the paper submission are 
1. ./stable_baselines/run_experiments.py
2. ./DartEnv2/gym/envs/dart/arm_acceleration_env_ball2.py
3. ./DartEnv2/gym/envs/dart/KR5_arm_acceleration.py
