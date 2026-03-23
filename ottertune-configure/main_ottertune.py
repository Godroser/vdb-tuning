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
VDTUNER_AUTO_CONFIGURE_PATH = os.path.abspath('/talas-pool/home/z78ding/vdb-tuning/auto-configure')
if VDTUNER_AUTO_CONFIGURE_PATH not in sys.path:
    sys.path.append(VDTUNER_AUTO_CONFIGURE_PATH)

# Import OtterTune and VDTuner modules
from analysis.gp_tf import GPRGD
from vdtuner_interface.utils import LHS_sample, KnobStand, RealEnv
from configure import configure_index, filter_index_rule, configure_system, filter_system_rule

# Configuration
KNOB_PATH = '/talas-pool/home/z78ding/vdb-tuning/auto-configure/whole_param.json'
RUN_ENGINE_PATH = '/talas-pool/home/z78ding/vdb-tuning/vector-db-benchmark-master/run_engine_test.sh'
USE_SUDO = True  # Set False if user has docker group (sudo usermod -aG docker $USER)

# Log file
LOG_FILE = 'glove-100-angular_new.log'

def run_engine_test(use_sudo=None):
    """Run the engine test and return performance metrics.
    use_sudo: Set False if user has docker group (no sudo needed). Defaults to USE_SUDO.
    Parsing logic follows auto-configure/vdtuner/utils.py RealEnv.get_state.
    """
    if use_sudo is None:
        use_sudo = USE_SUDO
    try:
        cmd = ["timeout", "900", RUN_ENGINE_PATH, "milvus-single-node", "milvus-p10", "glove-100-angular"]
        if use_sudo:
            cmd = ["sudo"] + cmd
        result = sp.run(
            cmd,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            cwd=os.path.dirname(RUN_ENGINE_PATH),
        )
        result_output = result.stdout.decode(errors='ignore') + result.stderr.decode(errors='ignore')
        lines = result_output.strip().split('\n')

        # Search backwards for result section (ref: auto-configure RealEnv.get_state)
        # Script outputs "📊 测试结果摘要:" then mean_precisions rps p95_time
        numeric_values = []
        for line in reversed(lines):
            if '测试结果摘要' in line or '📊' in line or '结果' in line:
                break
            for word in line.strip().split():
                try:
                    numeric_values.append(float(word))
                except ValueError:
                    continue

        # Fallback: extract all numeric values from entire output
        from_reversed = len(numeric_values) >= 3
        if not from_reversed:
            numeric_values = []
            for item in result_output.strip().split():
                try:
                    numeric_values.append(float(item))
                except ValueError:
                    continue

        if len(numeric_values) < 2:
            return None, None

        # Reversed: [p95_time, rps, mean_precisions]; Fallback: [..., mean_precisions, rps, p95_time]
        rps = numeric_values[-2]
        precision = numeric_values[-1] if from_reversed else (numeric_values[-3] if len(numeric_values) >= 3 else numeric_values[-1])

        if 0 <= precision <= 1 and rps > 0:
            return rps, precision
        return None, None
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
    X_success, y_success = [], []
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
            
            X_success.append(sample)
            y_success.append(objective)
            log(f"Performance: RPS = {rps:.4f}, Precision = {precision:.4f}, Objective = {objective:.4f}")
        else:
            log("Engine test failed, skipping this sample")
    
    X_train = np.array(X_success, dtype=np.float32) if X_success else np.zeros((0, num_knobs), dtype=np.float32)
    y_train = np.array(y_success, dtype=np.float32).reshape(-1, 1) if y_success else np.zeros((0, 1), dtype=np.float32)
    log(f"\nInitial training data shape: X={X_train.shape}, y={y_train.shape}")

    if len(X_success) == 0:
        log("ERROR: All initial samples failed. Check Docker permissions (add user to docker group) and ensure Milvus can start.")
        log("  Fix: sudo usermod -aG docker $USER  (then log out and back in)")
        return

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
    num_iterations = 250 - num_initial_samples
    
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
        
        # Select best candidate: use minl (predicted loss), not minl_conf (config values)
        # minl_conf is (n_candidates, n_feats) - argmin on it returns flattened index (wrong!)
        minl_flat = np.asarray(results.minl).ravel()
        best_idx = int(np.argmin(minl_flat))
        best_candidate = results.minl_conf[best_idx]  # Use optimized config from gradient descent
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
            if precision >= 0.8:
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
