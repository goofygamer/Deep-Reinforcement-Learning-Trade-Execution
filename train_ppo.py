import pandas as pd
from stable_baselines3 import PPO
from execution_env import ExecutionEnv
import argparse # We'll use this to pass in the run number

def main(run_id):
    print(f"--- Starting Training Run #{run_id} ---")
    
    print("--- Loading Data for Training ---")
    local_file_path = 'data/btc_us_1min_bars_2023-05-01_to_2023-05-31.csv'
    df = pd.read_csv(local_file_path, index_col='timestamp', parse_dates=True)
    
    print("--- Setting up PPO Training Environment ---")
    # Set a different random seed for each run to ensure variety
    env = ExecutionEnv(df=df, initial_inventory=50, trade_horizon=60)
    
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=f"./ppo_tensorboard_run_{run_id}/", seed=run_id)

    print(f"--- Training PPO Agent for Run #{run_id} ---")
    model.learn(total_timesteps=50000, progress_bar=True)
    print(f"--- Training Complete for Run #{run_id} ---")

    # Save the model with a unique name
    model_save_path = f"ppo_execution_agent_run_{run_id}"
    model.save(model_save_path)
    print(f"--- Trained PPO model saved to {model_save_path}.zip ---")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train PPO agent for optimal execution.")
    parser.add_argument('--runs', type=int, default=5, help='Number of independent training runs to perform.')
    args = parser.parse_args()

    for i in range(args.runs):
        main(run_id=i)