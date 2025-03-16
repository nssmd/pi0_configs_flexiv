#!/usr/bin/env python

# 设置使用第三张GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  
print(f"使用GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"JAX内存限制: {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")

import numpy as np
from openpi.training import config
from openpi.policies import policy_config
from openpi.policies.flexiv_policy import make_flexiv_example

# 设置检查点路径
CONFIG_NAME = "pi0_flexiv_low_mem_finetune"
CHECKPOINT_DIR = "/home/lenovo/zimo/pi0/openpi/checkpoints/pi0_flexiv_low_mem_finetune/flexiv_experiment/2000"

def main():
    print(f"加载配置: {CONFIG_NAME}")
    print(f"检查点路径: {CHECKPOINT_DIR}")
    
    # 加载配置
    train_config = config.get_config(CONFIG_NAME)
    
    # 打印配置信息
    print("\n配置信息:")
    print(f"模型类型: {train_config.model.__class__.__name__}")
    print(f"动作维度 (action_dim): {train_config.model.action_dim}")
    print(f"动作序列长度 (action_horizon): {train_config.model.action_horizon}")
    
    # 创建策略
    print("\n创建策略...")
    policy = policy_config.create_trained_policy(train_config, CHECKPOINT_DIR)
    
    # 创建测试输入
    print("\n创建测试输入...")
    example = make_flexiv_example()
    
    # 打印输入信息
    print(f"输入状态形状: {example['observation/state'].shape}")
    print(f"输入图像形状: {example['observation/image'].shape}")
    print(f"输入手腕图像形状: {example['observation/wrist_image'].shape}")
    print(f"输入提示: {example['prompt']}")
    
    # 运行推理
    print("\n运行推理...")
    result = policy.infer(example)
    
    # 打印输出信息
    print("\n推理结果:")
    if "actions" in result:
        actions = result["actions"]
        print(f"输出动作形状: {actions.shape}")
        print(f"输出动作类型: {type(actions)}")
        print(f"输出动作数据类型: {actions.dtype}")
        print("\n输出动作内容:")
        print(actions)
        
        # 如果是二维数组，打印每个时间步的动作维度
        if len(actions.shape) == 2:
            print("\n每个时间步的动作:")
            for i, action in enumerate(actions):
                print(f"时间步 {i}: {action.shape} - {action}")
    else:
        print("结果中没有找到'actions'键")
        print(f"可用的键: {list(result.keys())}")

if __name__ == "__main__":
    main() 