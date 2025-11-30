"""
Módulo de visualizações para a Calculadora PINN + Black-Scholes
Gera gráficos 2D e 3D comparativos e de análise
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm

# Configuração de estilo
plt.style.use('dark_background')

class Visualizador:
    """Gerencia a criação de gráficos e visualizações"""
    
    def __init__(self):
        self.cores = {
            'fundo': '#1e1e2e',
            'texto': '#ffffff',
            'primaria': '#64b5f6',
            'secundaria': '#ffb74d',
            'terciaria': '#81c784',
            'grid': '#333333'
        }
    
    def criar_figura(self, tamanho=(8, 6), dpi=100, 
                    projecao_3d=False):
        """Cria uma figura matplotlib configurada para o tema"""
        fig = Figure(figsize=tamanho, dpi=dpi)
        fig.patch.set_facecolor(self.cores['fundo'])
        
        if projecao_3d:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor(self.cores['fundo'])
            # Cores dos painéis 3D
            ax.w_xaxis.set_pane_color((0.15, 0.15, 0.2, 1.0))
            ax.w_yaxis.set_pane_color((0.15, 0.15, 0.2, 1.0))
            ax.w_zaxis.set_pane_color((0.15, 0.15, 0.2, 1.0))
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.cores['fundo'])
            ax.grid(True, linestyle='--', alpha=0.3, color=self.cores['grid'])
            
        # Configuração comum de eixos
        ax.tick_params(colors=self.cores['texto'])
        for spine in ax.spines.values():
            spine.set_color(self.cores['grid'])
            
        return fig, ax
    
    def plotar_comparacao_2d(self, S_range, precos_analiticos, precos_pinn, K):
        """Plota comparação 2D entre modelo analítico e PINN"""
        fig, ax = self.criar_figura()
        
        ax.plot(S_range, precos_analiticos, '-', 
               label='Black-Scholes (Analítico)', 
               color=self.cores['primaria'], linewidth=2)
               
        ax.plot(S_range, precos_pinn, '--', 
               label='PINN (Rede Neural)', 
               color=self.cores['secundaria'], linewidth=2)
        
        # Linha do Strike
        ax.axvline(x=K, color='red', linestyle=':', alpha=0.5, label='Strike (K)')
        
        ax.set_title('Comparação de Preços: Analítico vs PINN', 
                    color=self.cores['texto'], fontsize=14, pad=15)
        ax.set_xlabel('Preço do Ativo (S)', color=self.cores['texto'])
        ax.set_ylabel('Preço da Opção (V)', color=self.cores['texto'])
        
        legend = ax.legend(facecolor=self.cores['fundo'], edgecolor=self.cores['grid'])
        plt.setp(legend.get_texts(), color=self.cores['texto'])
        
        return fig
    
    def plotar_erro_absoluto(self, S_range, erro):
        """Plota o erro absoluto entre os modelos"""
        fig, ax = self.criar_figura()
        
        ax.plot(S_range, erro, color='#ef5350', linewidth=2)
        ax.fill_between(S_range, 0, erro, color='#ef5350', alpha=0.3)
        
        ax.set_title('Erro Absoluto (|Analítico - PINN|)', 
                    color=self.cores['texto'], fontsize=14)
        ax.set_xlabel('Preço do Ativo (S)', color=self.cores['texto'])
        ax.set_ylabel('Erro Absoluto', color=self.cores['texto'])
        
        return fig
    
    def plotar_superficie_3d(self, S_mesh, T_mesh, V_mesh, titulo='Superfície de Preço'):
        """Plota superfície 3D do preço da opção"""
        fig, ax = self.criar_figura(projecao_3d=True)
        
        surf = ax.plot_surface(S_mesh, T_mesh, V_mesh, 
                             cmap=cm.viridis,
                             linewidth=0, 
                             antialiased=True,
                             alpha=0.8)
        
        ax.set_title(titulo, color=self.cores['texto'], fontsize=14)
        ax.set_xlabel('Preço Ativo (S)', color=self.cores['texto'])
        ax.set_ylabel('Tempo (t)', color=self.cores['texto'])
        ax.set_zlabel('Valor Opção (V)', color=self.cores['texto'])
        
        # Barra de cores
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.ax.yaxis.set_tick_params(color=self.cores['texto'])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.cores['texto'])
        
        return fig
    
    def plotar_historico_treinamento(self, historico_loss):
        """Plota a evolução da perda durante o treinamento"""
        fig, ax = self.criar_figura()
        
        ax.plot(historico_loss, color=self.cores['terciaria'], linewidth=1.5)
        ax.set_yscale('log')
        
        ax.set_title('Convergência do Treinamento (Loss)', 
                    color=self.cores['texto'], fontsize=14)
        ax.set_xlabel('Época', color=self.cores['texto'])
        ax.set_ylabel('Loss (Log Scale)', color=self.cores['texto'])
        
        return fig
    
    def plotar_gregas(self, S_range, gregas_dict):
        """Plota as gregas em subplots"""
        fig = Figure(figsize=(10, 8), dpi=100)
        fig.patch.set_facecolor(self.cores['fundo'])
        
        gregas_nomes = ['delta', 'gamma', 'vega', 'theta']
        titulos = ['Delta (Δ)', 'Gamma (Γ)', 'Vega (ν)', 'Theta (Θ)']
        cores = ['#64b5f6', '#81c784', '#ffb74d', '#ba68c8']
        
        for i, (nome, titulo, cor) in enumerate(zip(gregas_nomes, titulos, cores)):
            ax = fig.add_subplot(2, 2, i+1)
            ax.set_facecolor(self.cores['fundo'])
            ax.grid(True, linestyle='--', alpha=0.3, color=self.cores['grid'])
            
            ax.plot(S_range, gregas_dict[nome], color=cor, linewidth=2)
            ax.set_title(titulo, color=self.cores['texto'])
            
            ax.tick_params(colors=self.cores['texto'])
            for spine in ax.spines.values():
                spine.set_color(self.cores['grid'])
                
        fig.tight_layout(pad=3.0)
        return fig
