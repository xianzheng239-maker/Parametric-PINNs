import numpy as np
import tensorflow as tf
import multiprocessing
from multiprocessing import Manager
import time

# import scipy.io as scio # Removido - não utilizado

# Definir parâmetros
sigmas1 = 0.002
sigmas2 = 0.008

# sigmaYs = 1 # Removido - pwnoise não é utilizado
# lambda_ = 1 # Removido - pwnoise não é utilizado
delta_t = 1
step_Num = 1000
T_max = step_Num * delta_t

alpha = -0.08
beta = 0.1


# Remover pwnoise, pois não estava sendo utilizado
# def pwnoise(...): ...

# Definir funções de deriva (drift)
def fun1(x2):
    return x2


# Renomeado para fun2_drift e modificado para NÃO incluir ruído
# O ruído será adicionado separadamente no integrador SDE
def fun2_drift(x1, x2, u):
    return -x1 + (alpha - beta * x1 ** 2) * x2 + u


# --- Inicializador do Worker para Multiprocessing ---
# Estas variáveis serão globais APENAS dentro de cada processo de trabalho
myControl = None
call_model_tf = None


def init_worker():
    """
    Função de inicialização para cada processo do pool.
    Carrega o modelo do TF e compila a função call_model
    APÓS o fork do processo.
    """
    global myControl, call_model_tf

    # 1. Carregar o modelo
    myControl = tf.saved_model.load('saved/NN/myControl')

    # 2. Definir e compilar a função de chamada do modelo
    @tf.function
    def call_model_internal(x1_val, x2_val):
        x1_val = tf.cast(x1_val, tf.float64)
        x2_val = tf.cast(x2_val, tf.float64)

        # CORREÇÃO: Usando x1_val em vez do valor fixo [[1.1]]
        # Isso assume que o primeiro input do modelo deve ser x1
        input_1 = x1_val

        # Esta era a linha não comentada no seu código
        input_2 = tf.concat([x1_val, x2_val], 1)

        return myControl([input_1, input_2])

    # Tornar a função compilada disponível para single_simulation
    call_model_tf = call_model_internal


# ----------------------------------------------------

# Remover definições globais de TF:
# myControl = tf.saved_model.load('saved/NN/myControl') # MOVIMENTADO PARA init_worker
# @tf.function def call_model(...): ... # MOVIMENTADO PARA init_worker


# Definir single_simulation
def single_simulation(j):
    """
    Tarefa de simulação única.
    Usa o método Euler-Maruyama para integração SDE.
    """
    # Gerar ruído Gaussiano
    # NOTA: O ruído agora é o incremento dW, que escala com sqrt(delta_t)
    dW1 = np.sqrt(sigmas1 * delta_t) * np.random.randn(step_Num)
    dW2 = np.sqrt(sigmas2 * delta_t) * np.random.randn(step_Num)

    # Ruído de Poisson (PWN) foi removido, pois não estava sendo usado.

    # Inicializar estado
    x1 = np.random.normal(0, np.sqrt(0.03))
    x2 = np.random.normal(0, np.sqrt(0.03))

    for i in range(step_Num):
        x1_tf = tf.constant([[x1]], dtype=tf.float64)
        x2_tf = tf.constant([[x2]], dtype=tf.float64)

        # Chama a função TF compilada carregada pelo worker
        # `call_model_tf` existe no escopo global deste worker
        u = call_model_tf(x1_tf, x2_tf).numpy()[0, 0]

        # --- CORREÇÃO: Método Euler-Maruyama ---

        # 1. Calcular a parte de deriva (drift)
        drift_x1 = fun1(x2)
        drift_x2 = fun2_drift(x1, x2, u)  # Usar a nova função fun2_drift

        # 2. Calcular a parte de difusão (ruído)
        # A SDE parece ser:
        # dx1 = fun1(x2) * dt
        # dx2 = fun2_drift(x1, x2, u) * dt + dW1 + dW2
        # (Se a intenção era sqrt(sigmas1)*dW1, ajuste dW1 acima)

        diffusion_x1 = 0.0  # Sem termo de ruído em x1
        diffusion_x2 = dW1[i] + dW2[i]

        # 3. Atualizar o estado
        x1 += drift_x1 * delta_t + diffusion_x1
        x2 += drift_x2 * delta_t + diffusion_x2
        # ----------------------------------------

    return x1, x2


# Função de simulação paralela
def parallel_simulation(num_simulations, num_processes):
    """
    Executa tarefas de simulação em paralelo.
    """
    # CORREÇÃO: Usar o `initializer` para carregar o modelo TF nos workers
    with multiprocessing.Pool(processes=num_processes,
                              initializer=init_worker) as pool:
        results = pool.map(single_simulation, range(num_simulations))

    final_values_X1, final_values_X2 = zip(*results)
    return np.array(final_values_X1), np.array(final_values_X2)


# Programa principal
if __name__ == "__main__":
    start_time = time.time()

    # Tarefas de simulação
    num_simulations = 1000  # Total de simulações
    num_processes = 10  # Processos paralelos

    # Executar simulação paralela
    final_values_X1, final_values_X2 = parallel_simulation(num_simulations, num_processes)

    # Salvar resultados
    A = np.column_stack((final_values_X1, final_values_X2))
    np.savetxt('pdf_001.txt', A, fmt='%.6f')

    # Calcular e imprimir o tempo total
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
