import sys
import signal
import time

sys.path.append("..")

from optimizer_pobo_sa import PollingBayesianOptimization
from utils import RealEnv


if __name__ == '__main__':
    # Log termination signals (e.g., logout/session cleanup) into stdout so they appear in `log/test.log`.
    def _handle_signal(signum, _frame):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"[{ts}] Received signal {signame} ({signum}). Exiting...", flush=True)
        raise SystemExit(128 + int(signum))

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            # Some signals may not be available/assignable on all platforms.
            pass

    # ============================================
    # 配置参数：在这里指定要测试的数据集
    # ============================================
    DATASET = "random-match-keyword-100-angular-no-filters"  # 可以修改为其他数据集，如 "glove-25-angular", "random-100" 等
    
    # prepare the environment
    env = RealEnv(dataset=DATASET)
    model = PollingBayesianOptimization(env, seed=1)
    
    # initial sampling
    model.init_sample()

    # iterative auto-tuning
    for i in range(200):
        model.step()
