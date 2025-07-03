# 模拟SD模型中加噪过程

import numpy as np
import matplotlib.pyplot as plt

def diffusion_process(x0, beta_values):
    """
        模拟前向扩散过程
        :param x0: 初始图像矩阵
        :param beta_values: 每一步的 beta 噪声系数列表
        :return: 所有步骤中的矩阵序列
    """
    steps = [x0]
    for beta_t in beta_values:
        prev = steps[-1]
        mean = np.sqrt(1 - beta_t) * prev
        noise = np.random.normal(0, np.sqrt(beta_t), size=prev.shape)
        xt = mean + noise
        steps.append(xt)
    return steps

def display_steps(matrices):
    """
        打印每一步的矩阵变化
    """
    for i, mat in enumerate(matrices):
        print(f"\nStep {i}:\n{np.round(mat, 4)}")

def plot_matrix_progression(matrices):
    """
        可视化每一步矩阵的数值变化
    """
    flattened = [mat.flatten() for mat in matrices]
    flattened = np.array(flattened)

    plt.figure(figsize=(8, 6))
    for i in range(flattened.shape[1]):
        plt.plot(flattened[:, i], marker='o', label=f'Element {i}')
    plt.title("Matrix Element Evolution Through Diffusion Steps")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 初始图像
    x0 = np.array([[1.0, 2.0],
                   [3.0, 4.0]])

    # beta_t 每步噪声强度，逐渐增大
    timesteps = 10
    beta_schedule = np.linspace(0.02, 0.15, timesteps)
    steps = diffusion_process(x0, beta_schedule)
    display_steps(steps)
    plot_matrix_progression(steps)
