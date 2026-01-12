import sys 
sys.path.append("..") 

from optimizer_pobo_sa import PollingBayesianOptimization
from utils import RealEnv


if __name__ == '__main__':
    # ============================================
    # 配置参数：在这里指定要测试的数据集
    # ============================================
    DATASET = "glove-100-angular"  # 可以修改为其他数据集，如 "glove-25-angular", "random-100" 等
    
    # prepare the environment
    env = RealEnv(dataset=DATASET)
    model = PollingBayesianOptimization(env, seed=1)
    
    # initial sampling
    model.init_sample()

    # iterative auto-tuning
    for i in range(200-7):
        model.step()
