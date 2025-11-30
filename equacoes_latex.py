"""
Sistema de renderização de equações LaTeX para a Calculadora PINN + Black-Scholes
"""
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# Configuração para renderização LaTeX de alta qualidade
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'


class GeradorEquacoes:
    """Classe para gerar e renderizar equações matemáticas"""
    
    def __init__(self):
        self.diretorio = os.path.join(os.path.dirname(__file__), 'equacoes')
        os.makedirs(self.diretorio, exist_ok=True)
        
        # Dicionário com todas as equações
        self.equacoes = {
            'black_scholes_pde': {
                'latex': r'$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0$',
                'titulo': 'Equação Diferencial Parcial de Black-Scholes',
                'descricao': 'EDP fundamental para precificação de opções'
            },
            'call_option': {
                'latex': r'$C(S,t) = SN(d_1) - Ke^{-r(T-t)}N(d_2)$',
                'titulo': 'Preço de Call Option',
                'descricao': 'Fórmula analítica para opção de compra'
            },
            'put_option': {
                'latex': r'$P(S,t) = Ke^{-r(T-t)}N(-d_2) - SN(-d_1)$',
                'titulo': 'Preço de Put Option',
                'descricao': 'Fórmula analítica para opção de venda'
            },
            'd1': {
                'latex': r'$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$',
                'titulo': 'Parâmetro d₁',
                'descricao': 'Primeiro parâmetro da distribuição normal'
            },
            'd2': {
                'latex': r'$d_2 = d_1 - \sigma\sqrt{T-t}$',
                'titulo': 'Parâmetro d₂',
                'descricao': 'Segundo parâmetro da distribuição normal'
            },
            'delta': {
                'latex': r'$\Delta = \frac{\partial V}{\partial S} = N(d_1)$',
                'titulo': 'Delta (Δ)',
                'descricao': 'Sensibilidade ao preço do ativo'
            },
            'gamma': {
                'latex': r'$\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{N^{\prime}(d_1)}{S\sigma\sqrt{T-t}}$',
                'titulo': 'Gamma (Γ)',
                'descricao': 'Taxa de variação do Delta'
            },
            'vega': {
                'latex': r'$\mathcal{V} = \frac{\partial V}{\partial \sigma} = S N^{\prime}(d_1)\sqrt{T-t}$',
                'titulo': 'Vega (ν)',
                'descricao': 'Sensibilidade à volatilidade'
            },
            'theta': {
                'latex': r'$\Theta = \frac{\partial V}{\partial t} = -\frac{SN^{\prime}(d_1)\sigma}{2\sqrt{T-t}} - rKe^{-r(T-t)}N(d_2)$',
                'titulo': 'Theta (Θ)',
                'descricao': 'Decaimento temporal da opção'
            },
            'rho': {
                'latex': r'$\rho = \frac{\partial V}{\partial r} = K(T-t)e^{-r(T-t)}N(d_2)$',
                'titulo': 'Rho (ρ)',
                'descricao': 'Sensibilidade à taxa de juros'
            },
            'funcao_perda_pinn': {
                'latex': r'$\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{BC} + \mathcal{L}_{IC}$',
                'titulo': 'Função de Perda da PINN',
                'descricao': 'Perda total combinando EDP, condições de contorno e iniciais'
            },
            'perda_pde': {
                'latex': r'$\mathcal{L}_{PDE} = \frac{1}{N}\sum_{i=1}^{N}\left|\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV\right|^2$',
                'titulo': 'Perda da EDP',
                'descricao': 'Erro na satisfação da equação de Black-Scholes'
            },
            'perda_bc': {
                'latex': r'$\mathcal{L}_{BC} = \frac{1}{N_{BC}}\sum_{i=1}^{N_{BC}}|V(S_i,t_i) - V_{BC}(S_i,t_i)|^2$',
                'titulo': 'Perda das Condições de Contorno',
                'descricao': 'Erro nas bordas do domínio'
            },
            'perda_ic': {
                'latex': r'$\mathcal{L}_{IC} = \frac{1}{N_{IC}}\sum_{i=1}^{N_{IC}}|V(S_i,T) - \max(S_i-K, 0)|^2$',
                'titulo': 'Perda das Condições Iniciais',
                'descricao': 'Erro no payoff final da opção'
            },
            'rede_neural': {
                'latex': r'$V_{PINN}(S,t) = NN(S,t;\theta)$',
                'titulo': 'Aproximação por Rede Neural',
                'descricao': 'PINN aproxima o preço da opção'
            },
        }
    
    def renderizar_equacao(self, nome_equacao, tamanho_figura=(10, 2), dpi=150):
        """Renderiza uma equação LaTeX e salva como imagem"""
        if nome_equacao not in self.equacoes:
            raise ValueError(f"Equação '{nome_equacao}' não encontrada")
        
        eq = self.equacoes[nome_equacao]
        
        fig, ax = plt.subplots(figsize=tamanho_figura, dpi=dpi)
        ax.axis('off')
        
        # Renderiza a equação
        ax.text(0.5, 0.5, eq['latex'], 
                fontsize=24, 
                ha='center', 
                va='center',
                color='white',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor='#1f538d', 
                         edgecolor='#64b5f6',
                         linewidth=2,
                         alpha=0.9))
        
        # Salva a imagem
        caminho = os.path.join(self.diretorio, f'{nome_equacao}.png')
        plt.tight_layout()
        plt.savefig(caminho, bbox_inches='tight', 
                   facecolor='none', 
                   edgecolor='none',
                   transparent=True)
        plt.close()
        
        return caminho
    
    def renderizar_equacao_com_titulo(self, nome_equacao, tamanho_figura=(12, 3), dpi=150):
        """Renderiza uma equação com título e descrição"""
        if nome_equacao not in self.equacoes:
            raise ValueError(f"Equação '{nome_equacao}' não encontrada")
        
        eq = self.equacoes[nome_equacao]
        
        fig, ax = plt.subplots(figsize=tamanho_figura, dpi=dpi)
        ax.axis('off')
        
        # Título
        ax.text(0.5, 0.85, eq['titulo'],
                fontsize=18,
                ha='center',
                va='top',
                color='#64b5f6',
                weight='bold')
        
        # Equação
        ax.text(0.5, 0.5, eq['latex'],
                fontsize=22,
                ha='center',
                va='center',
                color='white',
                bbox=dict(boxstyle='round,pad=0.8',
                         facecolor='#1f538d',
                         edgecolor='#64b5f6',
                         linewidth=2,
                         alpha=0.9))
        
        # Descrição
        ax.text(0.5, 0.15, eq['descricao'],
                fontsize=14,
                ha='center',
                va='bottom',
                color='#b0bec5',
                style='italic')
        
        # Salva a imagem
        caminho = os.path.join(self.diretorio, f'{nome_equacao}_completo.png')
        plt.tight_layout()
        plt.savefig(caminho, bbox_inches='tight',
                   facecolor='#1e1e2e',
                   edgecolor='none')
        plt.close()
        
        return caminho
    
    def renderizar_todas_equacoes(self):
        """Renderiza todas as equações disponíveis"""
        print("📐 Gerando equações LaTeX...")
        caminhos = {}
        
        for nome in self.equacoes.keys():
            # Versão simples
            caminho_simples = self.renderizar_equacao(nome)
            # Versão completa com título
            caminho_completo = self.renderizar_equacao_com_titulo(nome)
            
            caminhos[nome] = {
                'simples': caminho_simples,
                'completo': caminho_completo
            }
            print(f"  ✓ {self.equacoes[nome]['titulo']}")
        
        print(f"\n✅ Todas as equações foram salvas em: {self.diretorio}")
        return caminhos
    
    def criar_painel_explicativo(self):
        """Cria um painel com múltiplas equações organizadas"""
        fig = plt.figure(figsize=(14, 10), dpi=120)
        fig.patch.set_facecolor('#1e1e2e')
        
        # Título principal
        fig.suptitle('Modelo Black-Scholes e PINN', 
                    fontsize=28, 
                    color='#64b5f6',
                    weight='bold',
                    y=0.98)
        
        # Grid de subplots
        gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3,
                             left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        equacoes_ordem = [
            'black_scholes_pde', 'funcao_perda_pinn',
            'call_option', 'put_option',
            'd1', 'd2',
            'delta', 'gamma',
            'vega', 'theta'
        ]
        
        for idx, nome_eq in enumerate(equacoes_ordem):
            row = idx // 2
            col = idx % 2
            
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
            
            eq = self.equacoes[nome_eq]
            
            # Título da equação
            ax.text(0.5, 0.75, eq['titulo'],
                   fontsize=12,
                   ha='center',
                   va='top',
                   color='#64b5f6',
                   weight='bold',
                   transform=ax.transAxes)
            
            # Equação
            ax.text(0.5, 0.4, eq['latex'],
                   fontsize=14,
                   ha='center',
                   va='center',
                   color='white',
                   transform=ax.transAxes)
            
            # Descrição
            ax.text(0.5, 0.05, eq['descricao'],
                   fontsize=9,
                   ha='center',
                   va='bottom',
                   color='#90a4ae',
                   style='italic',
                   transform=ax.transAxes)
        
        # Salva o painel
        caminho = os.path.join(self.diretorio, 'painel_completo.png')
        plt.savefig(caminho, bbox_inches='tight',
                   facecolor='#1e1e2e',
                   edgecolor='none')
        plt.close()
        
        print(f"  ✓ Painel explicativo completo criado")
        return caminho
    
    def obter_info_equacao(self, nome_equacao):
        """Retorna informações sobre uma equação"""
        if nome_equacao in self.equacoes:
            return self.equacoes[nome_equacao]
        return None
    
    def listar_equacoes(self):
        """Lista todas as equações disponíveis"""
        return list(self.equacoes.keys())


if __name__ == "__main__":
    gerador = GeradorEquacoes()
    gerador.renderizar_todas_equacoes()
    gerador.criar_painel_explicativo()
    
    print("\n📋 Equações disponíveis:")
    for nome in gerador.listar_equacoes():
        info = gerador.obter_info_equacao(nome)
        print(f"  • {nome}: {info['titulo']}")
