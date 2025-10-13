from tqdm import tqdm
import time
from train.joint_ddp import Joint_train

if __name__ == '__main__':
    for i in tqdm(range(100), desc="Training Progress", ncols=100):
        time.sleep(0.01)  # 模拟耗时

    Joint_train()
