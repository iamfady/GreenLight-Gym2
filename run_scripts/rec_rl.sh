#!/bin/bash
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# Run the experiment manager with command line arguments
python gl_gym/RL/experiment_manager.py \
    --project AgriControl \
    --env_id TomatoEnv \
    --algorithm recurrentppo \
    --group rec-ppo-det \
    --n_eval_episodes 1 \
    --n_evals 1 \
    --env_seed 666 \
    --model_seed 666\
    --device cpu \
    --save_model \
    --save_env \
    # --hyperparameter_tuning
