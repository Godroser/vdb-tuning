#!/usr/bin/env python
import os
import sys
import numpy as np
import time
import subprocess as sp

# Add OtterTune server path to Python path
OTTERTUNE_SERVER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ottertune/server'))
if OTTERTUNE_SERVER_PATH not in sys.path:
    sys.path.append(OTTERTUNE_SERVER_PATH)

# Add VDTuner auto-configure path to Python path
VDTUNER_AUTO_CONFIGURE_PATH = os.path.abspath('/home/dyx/VDTuner/auto-configure')
if VDTUNER_AUTO_CONFIGURE_PATH not in sys.path:
    sys.path.append(VDTUNER_AUTO_CONFIGURE_PATH)

# Import OtterTune and VDTuner modules
from analysis.gp_tf import GPRGD
from vdtuner_interface.utils import LHS_sample, KnobStand, RealEnv
from configure import configure_index, filter_index_rule, configure_system, filter_system_rule

# Configuration
KNOB_PATH = '/home/dyx/VDTuner/auto-configure/whole_param.json'
RUN_ENGINE_PATH = '/home/dyx/VDTuner/vector-db-benchmark-master/run_engine_test.sh'

# Log file
LOG_FILE = 'ottertune_vdtuner.log'

def run_engine_test():
    """Run the engine test and return performance metrics"""
    try:
        result = sp.run(f'timeout 900 {RUN_ENGINE_PATH} "" "" glove-100-angular', shell=True, stdout=sp.PIPE)
        result = result.stdout.decode().split()
        rps = float(result[-2])
        precision = float(result[-3])
        return rps, precision
    except Exception as e:
        print(f"Error running engine test: {e}")
        return None, None

def log(message):
    """Log message to file and print"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')

def main():
    log("--- OtterTune VDTuner Integration ---\n")

    # 1. Initialize environment and knob stand
    log("Initializing environment...")
    env = RealEnv(bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH)
    knob_stand = KnobStand(KNOB_PATH)
    num_knobs = len(knob_stand.knobs_detail.keys())
    log(f"Number of knobs: {num_knobs}")

    # 2. Initial sampling using LHS
    log("\n--- Initial Sampling ---\n")
    num_initial_samples = 7
    log(f"Generating {num_initial_samples} initial samples using Latin Hypercube Sampling...")
    
    X_train = LHS_sample(num_knobs, num_initial_samples, seed=1)
    log(f"Initial samples shape: {X_train.shape}")

    # 3. Run initial samples to get performance data
    y_train = []
    for i, sample in enumerate(X_train):
        log(f"\nRunning initial sample {i+1}/{num_initial_samples}...")
        
        # Scale back to real values
        conf_values = [knob_stand.scale_back(env.names[j], sample[j])[1] for j in range(num_knobs)]
        index_values, system_values = conf_values[:9], conf_values[9:]
        index_names, system_names = env.names[:9], env.names[9:]
        
        index_conf = dict(zip(index_names, index_values))
        system_conf = dict(zip(system_names, system_values))
        
        log(f"Index configuration: {index_conf}")
        log(f"System configuration: {system_conf}")
        
        # Configure the system
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))
        
        # Run engine test
        rps, precision = run_engine_test()
        if rps is not None and precision is not None:
            # Single objective: maximize RPS when precision >= 0.9, else penalize
            if precision >= 0.9:
                objective = -rps  # We want to minimize, so negative RPS
            else:
                objective = 1e6  # Large penalty for low precision
            
            y_train.append(objective)
            log(f"Performance: RPS = {rps:.4f}, Precision = {precision:.4f}, Objective = {objective:.4f}")
        else:
            log("Engine test failed, skipping this sample")
            X_train = np.delete(X_train, i, axis=0)
    
    y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    log(f"\nInitial training data shape: X={X_train.shape}, y={y_train.shape}")

    # 4. Scale performance data
    y_min = np.min(y_train)
    y_max = np.max(y_train)
    y_train_scaled = (y_train - y_min) / (y_max - y_min) if y_max > y_min else y_train

    # 5. Initialize OtterTune GPRGD model
    log("\n--- Initializing OtterTune GPRGD Model ---\n")
    model = GPRGD(
        length_scale=1.0,
        magnitude=1.0,
        ridge=0.1,
        max_iter=20,
        learning_rate=0.001,
        check_numerics=True,
        debug=True
    )

    # 6. Run iterative tuning
    log("\n--- Iterative Tuning ---\n")
    num_iterations = 200 - num_initial_samples
    
    X_min = np.zeros(num_knobs, dtype=np.float32)
    X_max = np.ones(num_knobs, dtype=np.float32)
    
    for iteration in range(num_iterations):
        log(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        # Fit model to current data
        log("Fitting GPRGD model...")
        model.fit(X_train, y_train_scaled, X_min=X_min, X_max=X_max)
        
        # Generate new configuration candidates
        log("Generating new configuration candidates...")
        # Use LHS to generate candidate points
        num_candidates = 10
        X_candidates = LHS_sample(num_knobs, num_candidates, seed=iteration+2)
        
        # Predict objective for candidates
        log("Predicting objective for candidates...")
        results = model.predict(X_candidates)
        
        # Select best candidate (minimal objective)
        best_idx = np.argmin(results.minl_conf)
        best_candidate = X_candidates[best_idx]
        log(f"Best candidate found: {best_candidate}")
        
        # Scale back to real values
        conf_values = [knob_stand.scale_back(env.names[j], best_candidate[j])[1] for j in range(num_knobs)]
        index_values, system_values = conf_values[:9], conf_values[9:]
        index_names, system_names = env.names[:9], env.names[9:]
        
        index_conf = dict(zip(index_names, index_values))
        system_conf = dict(zip(system_names, system_values))
        
        log(f"Best index configuration: {index_conf}")
        log(f"Best system configuration: {system_conf}")
        
        # Configure the system
        configure_index(*filter_index_rule(index_conf))
        configure_system(filter_system_rule(system_conf))
        
        # Run engine test
        log("Running engine test with best candidate...")
        rps, precision = run_engine_test()
        
        if rps is not None and precision is not None:
            # Calculate objective
            if precision >= 0.9:
                objective = -rps
            else:
                objective = 1e6
            
            # Scale objective
            objective_scaled = (objective - y_min) / (y_max - y_min) if y_max > y_min else objective
            
            # Add to training data
            X_train = np.vstack((X_train, best_candidate.reshape(1, -1)))
            y_train = np.vstack((y_train, np.array([[objective]])))
            y_train_scaled = np.vstack((y_train_scaled, np.array([[objective_scaled]])))
            
            log(f"Performance: RPS = {rps:.4f}, Precision = {precision:.4f}, Objective = {objective:.4f}")
            log(f"Updated training data shape: X={X_train.shape}, y={y_train.shape}")
        else:
            log("Engine test failed, skipping this candidate")
        
        # Update min and max for scaling
        y_min = np.min(y_train)
        y_max = np.max(y_train)

    log("\n--- Tuning Complete ---")
    log(f"Final training data shape: X={X_train.shape}, y={y_train.shape}")

if __name__ == '__main__':
    # Clear log file
    with open(LOG_FILE, 'w') as f:
        f.write('')
    
    main()
