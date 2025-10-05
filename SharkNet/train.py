import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_environment import SharkGymEnv

MAPS_DIR = "processed_maps"
MODELS_DIR = "models"
LOGS_DIR = "logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TRAINING_TIMESTEPS = 50000

if __name__ == '__main__':
    print("Initialising environment...")
    # The environment will randomly load one of the maps for each episode
    env = SharkGymEnv(processed_maps_dir=MAPS_DIR)
    
    print("Creating PPO agent...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        device='auto' 
    )

    print(f"Starting training for {TRAINING_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TRAINING_TIMESTEPS,
        progress_bar=True
    )
    print("Training finished.")

    model_path = os.path.join(MODELS_DIR, f"ppo_shark_model.zip")
    model.save(model_path)

    print(f"Model saved to: {model_path}")
    print("\nTo visualize training progress, run the following command in your terminal:")
    print(f"tensorboard --logdir {LOGS_DIR}")

    env.close()