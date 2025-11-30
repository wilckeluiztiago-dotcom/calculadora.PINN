"""
Implementação do modelo Black-Scholes analítico
Inclui precificação de opções e cálculo das gregas
"""
import numpy as np
from scipy.stats import norm


class ModeloBlackScholes:
    """
    Modelo Black-Scholes para precificação de opções
    
    Parâmetros:
        preco_ativo (float): Preço atual do ativo subjacente (S)
        preco_strike (float): Preço de exercício (K)
        tempo_maturidade (float): Tempo até o vencimento em anos (T)
        taxa_juros (float): Taxa de juros livre de risco anual (r)
        volatilidade (float): Volatilidade do ativo (σ)
    """
    
    def __init__(self):
        self.preco_ativo = None
        self.preco_strike = None
        self.tempo_maturidade = None
        self.taxa_juros = None
        self.volatilidade = None
    
    def configurar_parametros(self, preco_ativo, preco_strike, tempo_maturidade, 
                             taxa_juros, volatilidade):
        """Define os parâmetros do modelo"""
        self.preco_ativo = float(preco_ativo)
        self.preco_strike = float(preco_strike)
        self.tempo_maturidade = float(tempo_maturidade)
        self.taxa_juros = float(taxa_juros)
        self.volatilidade = float(volatilidade)
        
        # Validação
        if self.preco_ativo <= 0:
            raise ValueError("Preço do ativo deve ser positivo")
        if self.preco_strike <= 0:
            raise ValueError("Preço strike deve ser positivo")
        if self.tempo_maturidade <= 0:
            raise ValueError("Tempo até maturidade deve ser positivo")
        if self.volatilidade <= 0:
            raise ValueError("Volatilidade deve ser positiva")
    
    def calcular_d1_d2(self):
        """
        Calcula os parâmetros d1 e d2 da fórmula de Black-Scholes
        
        d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
        d2 = d1 - σ√T
        """
        S = self.preco_ativo
        K = self.preco_strike
        T = self.tempo_maturidade
        r = self.taxa_juros
        sigma = self.volatilidade
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return d1, d2
    
    def calcular_preco_call(self):
        """
        Calcula o preço de uma opção de compra (call)
        
        C = S·N(d1) - K·e^(-rT)·N(d2)
        """
        d1, d2 = self.calcular_d1_d2()
        
        S = self.preco_ativo
        K = self.preco_strike
        T = self.tempo_maturidade
        r = self.taxa_juros
        
        preco_call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return preco_call
    
    def calcular_preco_put(self):
        """
        Calcula o preço de uma opção de venda (put)
        
        P = K·e^(-rT)·N(-d2) - S·N(-d1)
        """
        d1, d2 = self.calcular_d1_d2()
        
        S = self.preco_ativo
        K = self.preco_strike
        T = self.tempo_maturidade
        r = self.taxa_juros
        
        preco_put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return preco_put
    
    def calcular_delta(self, tipo_opcao='call'):
        """
        Calcula Delta: sensibilidade do preço da opção ao preço do ativo
        
        Delta_call = N(d1)
        Delta_put = N(d1) - 1
        """
        d1, _ = self.calcular_d1_d2()
        
        if tipo_opcao.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def calcular_gamma(self):
        """
        Calcula Gamma: taxa de variação do Delta
        
        Gamma = N'(d1) / (S·σ·√T)
        """
        d1, _ = self.calcular_d1_d2()
        
        S = self.preco_ativo
        sigma = self.volatilidade
        T = self.tempo_maturidade
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    def calcular_vega(self):
        """
        Calcula Vega: sensibilidade à volatilidade
        
        Vega = S·N'(d1)·√T
        """
        d1, _ = self.calcular_d1_d2()
        
        S = self.preco_ativo
        T = self.tempo_maturidade
        
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega / 100  # Dividido por 100 para representar mudança de 1%
    
    def calcular_theta(self, tipo_opcao='call'):
        """
        Calcula Theta: decaimento temporal da opção
        
        Theta_call = -S·N'(d1)·σ/(2√T) - r·K·e^(-rT)·N(d2)
        Theta_put = -S·N'(d1)·σ/(2√T) + r·K·e^(-rT)·N(-d2)
        """
        d1, d2 = self.calcular_d1_d2()
        
        S = self.preco_ativo
        K = self.preco_strike
        T = self.tempo_maturidade
        r = self.taxa_juros
        sigma = self.volatilidade
        
        termo1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if tipo_opcao.lower() == 'call':
            termo2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            termo2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        theta = termo1 + termo2
        return theta / 365  # Dividido por 365 para representar decaimento diário
    
    def calcular_rho(self, tipo_opcao='call'):
        """
        Calcula Rho: sensibilidade à taxa de juros
        
        Rho_call = K·T·e^(-rT)·N(d2)
        Rho_put = -K·T·e^(-rT)·N(-d2)
        """
        _, d2 = self.calcular_d1_d2()
        
        K = self.preco_strike
        T = self.tempo_maturidade
        r = self.taxa_juros
        
        if tipo_opcao.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
        return rho / 100  # Dividido por 100 para representar mudança de 1%
    
    def calcular_todas_gregas(self, tipo_opcao='call'):
        """Calcula todas as gregas de uma vez"""
        return {
            'delta': self.calcular_delta(tipo_opcao),
            'gamma': self.calcular_gamma(),
            'vega': self.calcular_vega(),
            'theta': self.calcular_theta(tipo_opcao),
            'rho': self.calcular_rho(tipo_opcao)
        }
    
    def calcular_superficie_precos(self, precos_ativos, tempos, tipo_opcao='call'):
        """
        Calcula superfície de preços para multiple valores de S e T
        
        Args:
            precos_ativos: Array de preços do ativo
            tempos: Array de tempos até maturidade
            tipo_opcao: 'call' ou 'put'
        
        Returns:
            Matriz de preços (len(tempos) x len(precos_ativos))
        """
        S_original = self.preco_ativo
        T_original = self.tempo_maturidade
        
        superficie = np.zeros((len(tempos), len(precos_ativos)))
        
        for i, t in enumerate(tempos):
            for j, s in enumerate(precos_ativos):
                self.preco_ativo = s
                self.tempo_maturidade = t
                
                if tipo_opcao.lower() == 'call':
                    superficie[i, j] = self.calcular_preco_call()
                else:
                    superficie[i, j] = self.calcular_preco_put()
        
        # Restaura valores originais
        self.preco_ativo = S_original
        self.tempo_maturidade = T_original
        
        return superficie
    
    def validar_paridade_put_call(self):
        """
        Verifica a paridade put-call: C - P = S - K·e^(-rT)
        
        Returns:
            Diferença entre os dois lados da equação (deve ser ~0)
        """
        preco_call = self.calcular_preco_call()
        preco_put = self.calcular_preco_put()
        
        lado_esquerdo = preco_call - preco_put
        lado_direito = (self.preco_ativo - 
                       self.preco_strike * np.exp(-self.taxa_juros * self.tempo_maturidade))
        
        diferenca = abs(lado_esquerdo - lado_direito)
        return diferenca
    
    def obter_resumo(self):
        """Retorna um resumo completo dos cálculos"""
        preco_call = self.calcular_preco_call()
        preco_put = self.calcular_preco_put()
        gregas_call = self.calcular_todas_gregas('call')
        gregas_put = self.calcular_todas_gregas('put')
        paridade = self.validar_paridade_put_call()
        
        return {
            'parametros': {
                'preco_ativo': self.preco_ativo,
                'preco_strike': self.preco_strike,
                'tempo_maturidade': self.tempo_maturidade,
                'taxa_juros': self.taxa_juros,
                'volatilidade': self.volatilidade
            },
            'precos': {
                'call': preco_call,
                'put': preco_put
            },
            'gregas_call': gregas_call,
            'gregas_put': gregas_put,
            'paridade_put_call': paridade
        }


# Funções auxiliares para uso direto
def calcular_preco_call(S, K, T, r, sigma):
    """Função auxiliar para calcular preço de call rapidamente"""
    modelo = ModeloBlackScholes()
    modelo.configurar_parametros(S, K, T, r, sigma)
    return modelo.calcular_preco_call()


def calcular_preco_put(S, K, T, r, sigma):
    """Função auxiliar para calcular preço de put rapidamente"""
    modelo = ModeloBlackScholes()
    modelo.configurar_parametros(S, K, T, r, sigma)
    return modelo.calcular_preco_put()


if __name__ == "__main__":
    # Teste do modelo
    print("🧮 Testando Modelo Black-Scholes\n")
    
    modelo = ModeloBlackScholes()
    modelo.configurar_parametros(
        preco_ativo=100,
        preco_strike=100,
        tempo_maturidade=1.0,
        taxa_juros=0.05,
        volatilidade=0.2
    )
    
    resumo = modelo.obter_resumo()
    
    print("Parâmetros:")
    for chave, valor in resumo['parametros'].items():
        print(f"  {chave}: {valor}")
    
    print("\nPreços:")
    print(f"  Call: R$ {resumo['precos']['call']:.4f}")
    print(f"  Put:  R$ {resumo['precos']['put']:.4f}")
    
    print("\nGregas (Call):")
    for chave, valor in resumo['gregas_call'].items():
        print(f"  {chave}: {valor:.6f}")
    
    print(f"\nParidade Put-Call (erro): {resumo['paridade_put_call']:.10f}")
    print("✅ Modelo validado!" if resumo['paridade_put_call'] < 1e-10 else "⚠️  Verificar paridade")
