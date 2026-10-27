import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Manager
import time
import scipy.io as scio



# 定义参数
sigmas1 = 0.002
sigmas2 = 0.008

sigmaYs = 1
lambda_ = 1
delta_t = 1
step_Num = 1000
T_max = step_Num * delta_t


alpha=-0.08
beta=0.1



# 定义泊松白噪声生成函数
def pwnoise(step_num, T, step, sigmaYs, lambda_):
    pw = np.zeros(step_num + 1)
    pt = 0
    Ynum = 1
    i = 1
    while pt <= T and i <= step_num:
        k = int(np.floor(pt / step))
        t1, t2 = np.random.random(), np.random.random()
        temp1 = np.sqrt(-2 * np.log(t1))
        temp2 = 2 * np.pi * t2
        p_amplitude = temp1 * (np.cos(temp2) if Ynum == 1 else np.sin(temp2))
        p_amplitude *= np.sqrt(sigmaYs)
        if k < len(pw):
            pw[k] += p_amplitude / step
        i += 1
        pt -= np.log(np.random.random()) / lambda_
    return pw

# 定义函数 fun1 和 fun2
def fun1(x2):
    return x2

def fun2(x1, x2, gnoise1, gnoise2, u):


    return -x1+(alpha-beta*x1**2)*x2+u+gnoise1+gnoise2

# 加载模型并创建一个被tf.function包装的调用函数
myControl = tf.saved_model.load('saved/NN/myControl')


@tf.function
def call_model(x1_val, x2_val):
    x1_val = tf.cast(x1_val, tf.float64)
    x2_val = tf.cast(x2_val, tf.float64)

    input_1=tf.constant([[1.1]], dtype=tf.float64)
    input_2 = tf.concat([x1_val, x2_val], 1)

    # 调整形状以匹配模型期望
    # 第一个输入：形状 (None, 1)
    #input_1 = tf.reshape(x1_val, [-1, 1])
    #input_2 = tf.stack([x2_val, tf.zeros_like(x2_val)], axis=1)
    #input_2 = tf.reshape(input_2, [-1, 2])
    return myControl([input_1, input_2])

# 定义单次模拟函数
def single_simulation(j):
    """
    单次模拟任务。
    """
    # Generate Gaussian white noise and Poisson white noise
    GWN1 = np.sqrt(sigmas1 / delta_t) * np.random.randn(step_Num + 1)
    GWN2 = np.sqrt(sigmas2 / delta_t) * np.random.randn(step_Num + 1)
    PWN = pwnoise(step_Num, T_max, delta_t, sigmaYs, lambda_)

    # Initialize state
    x1 = np.random.normal(0, np.sqrt(0.03))
    x2 = np.random.normal(0, np.sqrt(0.03))

    for i in range(step_Num):
        par=tf.constant([1.0], dtype=tf.float64)
        x1_tf = tf.constant([[x1]], dtype=tf.float64)
        x2_tf = tf.constant([[x2]], dtype=tf.float64)
        x_tf=[x1_tf,x2_tf]

        #input_control = tf.concat([x_tf, par], axis=1)
        u = call_model(x1_tf,x2_tf).numpy()[0, 0]

        k11 = fun1(x2)
        k21 = fun2(x1, x2, GWN1[i], GWN2[i], u)
        temp1 = x1 + 0.5 * delta_t * k11
        temp2 = x2 + 0.5 * delta_t * k21

        k12 = fun1(temp2)
        k22 = fun2(temp1, temp2, (GWN1[i] + GWN1[i + 1]) / 2, (GWN2[i] + GWN2[i + 1]) / 2, u)
        temp1 = x1 + 0.5 * delta_t * k12
        temp2 = x2 + 0.5 * delta_t * k22

        k13 = fun1(temp2)
        k23 = fun2(temp1, temp2, (GWN1[i] + GWN1[i + 1]) / 2,(GWN2[i] + GWN2[i + 1]) / 2,  u)
        temp1 = x1 + 0.5 * delta_t * k13
        temp2 = x2 + 0.5 * delta_t * k23

        k14 = fun1(temp2)
        k24 = fun2(temp1, temp2, (GWN1[i] + GWN1[i + 1]) / 2,(GWN2[i] + GWN2[i + 1]) / 2,  u)

        x1 += delta_t * (k11 + 2.0 * k12 + 2.0 * k13 + k14) / 6.0
        x2 += delta_t * (k21 + 2.0 * k22 + 2.0 * k23 + k24) / 6.0

    return x1, x2

# 并行模拟函数
def parallel_simulation(num_simulations, num_processes):
    """
    并行运行模拟任务。
    """
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(single_simulation, range(num_simulations))
    final_values_X1, final_values_X2 = zip(*results)
    return np.array(final_values_X1), np.array(final_values_X2)

# 主程序
if __name__ == "__main__":
    start_time = time.time()

    # 模拟任务
    num_simulations = 50000  # 模拟总次数
    num_processes = 10  # 并行进程数

    # 运行并行模拟
    final_values_X1, final_values_X2 = parallel_simulation(num_simulations, num_processes)

    # 保存结果
    A = np.column_stack((final_values_X1, final_values_X2))
    np.savetxt('pdf_001.txt', A, fmt='%.6f')

    # 计算并打印总时间
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
