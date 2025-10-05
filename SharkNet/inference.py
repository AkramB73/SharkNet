import os
import json
import numpy as np
from stable_baselines3 import PPO
from rl_environment import SharkGymEnv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MAPS_DIR = "processed_maps"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "ppo_shark_model.zip")
OUTPUT_DIR = "visualizer_logs"

TOTAL_SIMULATION_STEPS = 200

def plot_trajectory(log_data, output_path):
    history = log_data['history']
    environment = log_data['environment']
    chl_data = np.array(environment['chl_data'])
    x_coords = [step['position'][0] for step in history]
    y_coords = [step['position'][1] for step in history]

    fig, ax = plt.subplots(figsize=(10, 10))
    water_mask = (chl_data > 0).astype(int)
    cmap = mcolors.ListedColormap(['saddlebrown', 'deepskyblue'])
    ax.imshow(water_mask, cmap=cmap, origin='lower', interpolation='nearest')
    
    ax.plot(x_coords, y_coords, color='red', linewidth=2, label='Shark Trajectory')

    ax.plot(x_coords, y_coords, 'o', markersize=2, color='darkred', label='_nolegend_') 

    if x_coords:
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
        ax.plot(x_coords[-1], y_coords[-1], 'kx', markersize=10, label='End')

    ax.set_title("Shark Agent Trajectory")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {output_path}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = SharkGymEnv(processed_maps_dir=MAPS_DIR)
    model = PPO.load(MODEL_PATH, env=env)
    print(f"Model {MODEL_PATH} loaded successfully.")

    print(f"\nRunning a single simulation for {TOTAL_SIMULATION_STEPS} steps...")
    
    obs, info = env.reset()
    
    simulation_history = []
    env_data = {
        "width": env.model.width,
        "height": env.model.height,
        "chl_data": env.model.food_layer.tolist()
    }

    for current_step in range(TOTAL_SIMULATION_STEPS):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        agent_state = env.model.agents[0]
        
        simulation_history.append({
            "step": current_step,
            "position": [float(p) for p in agent_state.pos_float],
            "energy": float(agent_state.energy),
            "reward": float(reward),
            "done": bool(done)
        })

        if done:
            print(f"Simulation ended early at step {current_step} because the agent ran out of energy.")
            break
    
    final_log = {
        "simulation_summary": {
            "total_steps": len(simulation_history)
        },
        "environment": env_data,
        "history": simulation_history
    }

    log_path = os.path.join(OUTPUT_DIR, "simulation_log.json")
    with open(log_path, 'w') as f:
        json.dump(final_log, f, indent=4)

    print("\n--- Simulation Finished ---")
    print(f"Total Steps Executed: {len(simulation_history)}")
    print(f"Visualizer log saved to: {log_path}")
    
    plot_output_path = os.path.join(OUTPUT_DIR, "simulation_trajectory.png")
    plot_trajectory(final_log, plot_output_path)

    env.close()