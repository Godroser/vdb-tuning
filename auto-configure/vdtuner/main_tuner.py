import sys 
sys.path.append("..") 

from optimizer_pobo_sa import PollingBayesianOptimization
from utils import RealEnv


if __name__ == '__main__':
    # prepare the environment
    # Change dataset_name to use different datasets: "glove-100-angular", "deep-image-96-angular", etc.
    dataset_name = "random-match-keyword-100-angular-filters"  # Change this to your desired dataset
    env = RealEnv(dataset_name=dataset_name)
    model = PollingBayesianOptimization(env, seed=1)
    
    # initial sampling
    model.init_sample()

    # iterative auto-tuning
    for i in range(200):
        model.step()
