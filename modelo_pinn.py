"""
Implementação da Rede Neural Informada pela Física (PINN) para Black-Scholes
Utiliza TensorFlow para criar uma rede neural profunda que aprende a resolver a EDP
"""
import tensorflow as tf
import numpy as np
import time

# Configurações para melhor performance e reprodutibilidade
tf.random.set_seed(42)
np.random.seed(42)

class RedeNeuralPINN:
    """
    Rede Neural Informada pela Física para resolver a equação de Black-Scholes
    
    A rede aprende a função V(S, t) que satisfaz:
    1. A EDP de Black-Scholes no domínio
    2. A condição terminal (Payoff) em t=T
    3. As condições de contorno em S=0 e S=S_max
    """
    
    def __init__(self, camadas=[2, 50, 50, 50, 50, 1], learning_rate=0.001):
        self.camadas = camadas
        self.learning_rate = learning_rate
        self.modelo = self._construir_modelo()
        self.otimizador = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Histórico de perdas
        self.historico_loss = []
        self.historico_loss_pde = []
        self.historico_loss_bc = []
        self.historico_loss_ic = []
        
        # Parâmetros da equação (serão definidos no treinamento)
        self.r = None
        self.sigma = None
        self.K = None
        self.T = None
        
    def _construir_modelo(self):
        """Constrói a arquitetura da rede neural"""
        modelo = tf.keras.Sequential()
        
        # Camada de entrada (S, t)
        modelo.add(tf.keras.layers.InputLayer(input_shape=(2,)))
        
        # Camada de normalização (importante para PINNs)
        modelo.add(tf.keras.layers.Lambda(lambda x: 2.0 * (x - 0.0) / (1.0 - 0.0) - 1.0))
        
        # Camadas ocultas densas com ativação tanh (suave e diferenciável)
        for neuronios in self.camadas[1:-1]:
            modelo.add(tf.keras.layers.Dense(
                neuronios, 
                activation='tanh',
                kernel_initializer='glorot_normal'
            ))
        
        # Camada de saída (V - Preço da Opção)
        # Ativação softplus garante que o preço seja positivo
        modelo.add(tf.keras.layers.Dense(1, activation='softplus'))
        
        return modelo
    
    @tf.function
    def calcular_derivadas(self, S, t):
        """Calcula as derivadas necessárias para a EDP usando diferenciação automática"""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(S)
            tape2.watch(t)
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(S)
                tape1.watch(t)
                
                # Input concatenado para o modelo
                inputs = tf.stack([S, t], axis=1)
                V = self.modelo(inputs)
            
            # Primeiras derivadas
            dV_dS = tape1.gradient(V, S)
            dV_dt = tape1.gradient(V, t)
            
        # Segunda derivada
        d2V_dS2 = tape2.gradient(dV_dS, S)
        
        return V, dV_dt, dV_dS, d2V_dS2
    
    @tf.function
    def calcular_perda_pde(self, S, t):
        """
        Calcula o resíduo da EDP de Black-Scholes:
        Resíduo = ∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + r*S*∂V/∂S - r*V
        """
        V, dV_dt, dV_dS, d2V_dS2 = self.calcular_derivadas(S, t)
        
        # Equação de Black-Scholes
        residuo = dV_dt + 0.5 * (self.sigma**2) * (S**2) * d2V_dS2 + \
                  self.r * S * dV_dS - self.r * V
                  
        return tf.reduce_mean(tf.square(residuo))
    
    @tf.function
    def calcular_perda(self, S_colocacao, t_colocacao, 
                       S_inicial, t_inicial, V_inicial,
                       S_contorno_inf, t_contorno_inf, V_contorno_inf,
                       S_contorno_sup, t_contorno_sup, V_contorno_sup):
        """Calcula a perda total combinada"""
        
        # 1. Perda da EDP (Physics Loss) nos pontos de colocação
        loss_pde = self.calcular_perda_pde(S_colocacao, t_colocacao)
        
        # 2. Perda da Condição Inicial/Terminal (Payoff em t=T)
        # Nota: Em BS, a condição "inicial" para a EDP é no vencimento T
        inputs_inicial = tf.stack([S_inicial, t_inicial], axis=1)
        V_pred_inicial = self.modelo(inputs_inicial)
        loss_ic = tf.reduce_mean(tf.square(V_inicial - V_pred_inicial))
        
        # 3. Perda das Condições de Contorno (S=0 e S=Smax)
        inputs_inf = tf.stack([S_contorno_inf, t_contorno_inf], axis=1)
        V_pred_inf = self.modelo(inputs_inf)
        loss_bc_inf = tf.reduce_mean(tf.square(V_contorno_inf - V_pred_inf))
        
        inputs_sup = tf.stack([S_contorno_sup, t_contorno_sup], axis=1)
        V_pred_sup = self.modelo(inputs_sup)
        loss_bc_sup = tf.reduce_mean(tf.square(V_contorno_sup - V_pred_sup))
        
        loss_bc = loss_bc_inf + loss_bc_sup
        
        # Perda Total Ponderada
        # Pesos podem ser ajustados para forçar melhor as condições de contorno
        loss_total = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc
        
        return loss_total, loss_pde, loss_ic, loss_bc
    
    @tf.function
    def passo_treinamento(self, *args):
        """Executa um passo de otimização"""
        with tf.GradientTape() as tape:
            loss_total, loss_pde, loss_ic, loss_bc = self.calcular_perda(*args)
            
        gradientes = tape.gradient(loss_total, self.modelo.trainable_variables)
        self.otimizador.apply_gradients(zip(gradientes, self.modelo.trainable_variables))
        
        return loss_total, loss_pde, loss_ic, loss_bc

    def gerar_dados_treinamento(self, N_colocacao, N_borda, S_max):
        """Gera pontos aleatórios para treinamento"""
        # Pontos de Colocação (Domínio interno)
        S_col = tf.random.uniform((N_colocacao,), minval=0.1, maxval=S_max, dtype=tf.float32)
        t_col = tf.random.uniform((N_colocacao,), minval=0, maxval=self.T, dtype=tf.float32)
        
        # Condição Terminal (t=T, Payoff)
        S_ic = tf.random.uniform((N_borda,), minval=0, maxval=S_max, dtype=tf.float32)
        t_ic = tf.ones((N_borda,), dtype=tf.float32) * self.T
        V_ic = tf.maximum(S_ic - self.K, 0)  # Payoff Call Option
        V_ic = tf.reshape(V_ic, (-1, 1))
        
        # Condição de Contorno Inferior (S=0)
        S_bc_inf = tf.zeros((N_borda,), dtype=tf.float32)
        t_bc_inf = tf.random.uniform((N_borda,), minval=0, maxval=self.T, dtype=tf.float32)
        V_bc_inf = tf.zeros((N_borda, 1), dtype=tf.float32) # Call vale 0 se S=0
        
        # Condição de Contorno Superior (S=S_max)
        S_bc_sup = tf.ones((N_borda,), dtype=tf.float32) * S_max
        t_bc_sup = tf.random.uniform((N_borda,), minval=0, maxval=self.T, dtype=tf.float32)
        # V ≈ S - K*e^(-r(T-t)) para S muito grande (Call)
        V_bc_sup = S_bc_sup - self.K * tf.exp(-self.r * (self.T - t_bc_sup))
        V_bc_sup = tf.reshape(V_bc_sup, (-1, 1))
        
        return (S_col, t_col, S_ic, t_ic, V_ic, 
                S_bc_inf, t_bc_inf, V_bc_inf, 
                S_bc_sup, t_bc_sup, V_bc_sup)

    def treinar(self, K, T, r, sigma, epocas=2000, N_colocacao=5000, N_borda=500, callback_progresso=None):
        """
        Treina a PINN para os parâmetros especificados
        
        Args:
            K: Strike price
            T: Tempo de maturidade
            r: Taxa livre de risco
            sigma: Volatilidade
            epocas: Número de iterações de treinamento
            callback_progresso: Função para atualizar barra de progresso na GUI
        """
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        
        S_max = 4.0 * K  # Domínio espacial suficiente
        
        # Gera dados (fixos para todo o treinamento ou reamostrados periodicamente)
        dados = self.gerar_dados_treinamento(N_colocacao, N_borda, S_max)
        
        inicio = time.time()
        
        for epoca in range(epocas):
            loss_total, loss_pde, loss_ic, loss_bc = self.passo_treinamento(*dados)
            
            # Registra histórico
            if epoca % 10 == 0:
                self.historico_loss.append(loss_total.numpy())
                self.historico_loss_pde.append(loss_pde.numpy())
                self.historico_loss_ic.append(loss_ic.numpy())
                self.historico_loss_bc.append(loss_bc.numpy())
                
                # Callback para GUI
                if callback_progresso:
                    progresso = (epoca + 1) / epocas
                    msg = f"Época {epoca}/{epocas} - Loss: {loss_total:.6f}"
                    callback_progresso(progresso, msg)
                
                # Log no console ocasionalmente
                if epoca % 500 == 0:
                    print(f"Época {epoca}: Loss Total = {loss_total:.6f} (PDE={loss_pde:.6f}, IC={loss_ic:.6f}, BC={loss_bc:.6f})")
        
        tempo_total = time.time() - inicio
        print(f"Treinamento concluído em {tempo_total:.2f} segundos.")
        return self.historico_loss

    def prever(self, S, t):
        """Faz previsões usando o modelo treinado"""
        S = tf.convert_to_tensor(S, dtype=tf.float32)
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        
        # Garante formato correto
        if len(S.shape) == 0: S = tf.reshape(S, (1,))
        if len(t.shape) == 0: t = tf.reshape(t, (1,))
        
        # Se t for escalar e S vetor, expande t
        if t.shape[0] == 1 and S.shape[0] > 1:
            t = tf.ones_like(S) * t
            
        inputs = tf.stack([S, t], axis=1)
        return self.modelo(inputs).numpy().flatten()


if __name__ == "__main__":
    # Teste rápido
    print("🧠 Inicializando PINN...")
    pinn = RedeNeuralPINN(camadas=[2, 20, 20, 1], learning_rate=0.01)
    
    print("🏋️ Começando treinamento de teste...")
    historico = pinn.treinar(K=100, T=1.0, r=0.05, sigma=0.2, epocas=500)
    
    print("🔮 Fazendo previsão...")
    preco = pinn.prever(S=100, t=0)
    print(f"Preço previsto para S=100, t=0: {preco[0]:.4f}")
