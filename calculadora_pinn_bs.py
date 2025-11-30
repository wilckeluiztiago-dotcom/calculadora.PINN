import customtkinter as ctk
from tkinter import messagebox
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import os

# Importações locais
from black_scholes import ModeloBlackScholes
from modelo_pinn import RedeNeuralPINN
from visualizacoes import Visualizador
from equacoes_latex import GeradorEquacoes
import icones

# Configuração do CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class CalculadoraPINNApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configurações da Janela
        self.title("Calculadora PINN + Black-Scholes")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        # Inicialização dos Modelos
        self.modelo_bs = ModeloBlackScholes()
        self.pinn = None
        self.visualizador = Visualizador()
        self.gerador_eq = GeradorEquacoes()
        
        # Variáveis de Estado
        self.treinamento_em_andamento = False
        self.pinn_treinada = False
        
        # Configuração do Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self._criar_menu_lateral()
        self._criar_area_principal()
        
        # Carrega assets iniciais
        self.after(100, self._carregar_assets)

    def _carregar_assets(self):
        """Carrega ícones e equações"""
        try:
            # Garante que os diretórios existem
            icones.criar_todos_icones()
            self.gerador_eq.renderizar_todas_equacoes()
            self.gerador_eq.criar_painel_explicativo()
            
            # Carrega imagem do painel na aba Modelo
            self._atualizar_aba_modelo()
        except Exception as e:
            print(f"Erro ao carregar assets: {e}")

    def _criar_menu_lateral(self):
        """Cria o menu lateral de navegação"""
        self.frame_menu = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.frame_menu.grid(row=0, column=0, sticky="nsew")
        self.frame_menu.grid_rowconfigure(6, weight=1)

        # Título / Logo
        self.lbl_logo = ctk.CTkLabel(self.frame_menu, text="PINN\nFinance", 
                                   font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_logo.grid(row=0, column=0, padx=20, pady=20)

        # Botões de Navegação
        self.btn_modelo = self._criar_botao_menu("Modelo Matemático", self._mostrar_modelo, 1)
        self.btn_calc = self._criar_botao_menu("Calculadora", self._mostrar_calculadora, 2)
        self.btn_pinn = self._criar_botao_menu("Rede Neural (PINN)", self._mostrar_pinn, 3)
        self.btn_vis = self._criar_botao_menu("Visualizações", self._mostrar_visualizacoes, 4)
        
        # Info Footer
        self.lbl_footer = ctk.CTkLabel(self.frame_menu, text="v1.0.0\nPowered by TensorFlow",
                                     text_color="gray50", font=ctk.CTkFont(size=10))
        self.lbl_footer.grid(row=7, column=0, padx=20, pady=20)

    def _criar_botao_menu(self, texto, comando, row):
        btn = ctk.CTkButton(self.frame_menu, text=texto, command=comando,
                           fg_color="transparent", text_color=("gray10", "gray90"),
                           hover_color=("gray70", "gray30"), anchor="w", height=40)
        btn.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
        return btn

    def _criar_area_principal(self):
        """Cria a área onde o conteúdo das abas será exibido"""
        self.frame_conteudo = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frame_conteudo.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Cria as frames de cada aba
        self.frames = {}
        for nome in ["modelo", "calculadora", "pinn", "visualizacoes"]:
            frame = ctk.CTkFrame(self.frame_conteudo, corner_radius=10)
            self.frames[nome] = frame
            
        # Inicializa conteúdo de cada aba
        self._setup_aba_modelo()
        self._setup_aba_calculadora()
        self._setup_aba_pinn()
        self._setup_aba_visualizacoes()
        
        # Mostra a primeira aba
        self._mostrar_modelo()

    # --- Setup das Abas ---

    def _setup_aba_modelo(self):
        frame = self.frames["modelo"]
        frame.grid_columnconfigure(0, weight=1)
        
        lbl_titulo = ctk.CTkLabel(frame, text="Modelo Matemático: Black-Scholes & PINNs", 
                                font=ctk.CTkFont(size=24, weight="bold"))
        lbl_titulo.pack(pady=20)
        
        # Container para imagem das equações com scroll
        self.scroll_modelo = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        self.scroll_modelo.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.lbl_img_equacoes = ctk.CTkLabel(self.scroll_modelo, text="Carregando equações...")
        self.lbl_img_equacoes.pack(pady=10)
        
        texto_explicativo = """
        Este software utiliza uma abordagem híbrida para precificação de opções:
        
        1. Modelo Analítico de Black-Scholes:
           A solução clássica e exata para opções Europeias sob premissas ideais.
           
        2. Physics-Informed Neural Networks (PINNs):
           Uma rede neural profunda treinada não apenas com dados, mas com a própria
           Equação Diferencial Parcial (EDP) de Black-Scholes como função de perda.
           Isso permite que a rede aprenda a física do problema financeiro.
        """
        lbl_texto = ctk.CTkLabel(self.scroll_modelo, text=texto_explicativo, justify="left",
                               font=ctk.CTkFont(size=14))
        lbl_texto.pack(pady=20)

    def _setup_aba_calculadora(self):
        frame = self.frames["calculadora"]
        frame.grid_columnconfigure((0, 1), weight=1)
        
        # Painel de Entrada
        frame_input = ctk.CTkFrame(frame)
        frame_input.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(frame_input, text="Parâmetros de Entrada", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        self.entradas_calc = {}
        params = [
            ("Preço do Ativo (S)", "100.0"),
            ("Preço Strike (K)", "100.0"),
            ("Tempo Maturidade (T anos)", "1.0"),
            ("Taxa de Juros (r)", "0.05"),
            ("Volatilidade (σ)", "0.2")
        ]
        
        for label, valor_padrao in params:
            container = ctk.CTkFrame(frame_input, fg_color="transparent")
            container.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(container, text=label).pack(side="left")
            entry = ctk.CTkEntry(container, width=100)
            entry.insert(0, valor_padrao)
            entry.pack(side="right")
            self.entradas_calc[label] = entry
            
        btn_calcular = ctk.CTkButton(frame_input, text="Calcular Preço", 
                                   command=self._calcular_black_scholes,
                                   height=40, font=ctk.CTkFont(weight="bold"))
        btn_calcular.pack(pady=20, padx=10, fill="x")
        
        # Painel de Resultados
        frame_result = ctk.CTkFrame(frame)
        frame_result.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(frame_result, text="Resultados (Analítico)", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
        
        self.lbl_resultado_call = ctk.CTkLabel(frame_result, text="Call: ---", 
                                             font=ctk.CTkFont(size=20))
        self.lbl_resultado_call.pack(pady=10)
        
        self.lbl_resultado_put = ctk.CTkLabel(frame_result, text="Put: ---", 
                                            font=ctk.CTkFont(size=20))
        self.lbl_resultado_put.pack(pady=10)
        
        # Gregas
        self.frame_gregas = ctk.CTkFrame(frame_result, fg_color="transparent")
        self.frame_gregas.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.lbls_gregas = {}
        for grega in ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']:
            lbl = ctk.CTkLabel(self.frame_gregas, text=f"{grega}: ---")
            lbl.pack(anchor="w")
            self.lbls_gregas[grega] = lbl

    def _setup_aba_pinn(self):
        frame = self.frames["pinn"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        # Painel de Controle
        painel_controle = ctk.CTkFrame(frame)
        painel_controle.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        ctk.CTkLabel(painel_controle, text="Treinamento da Rede Neural", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=20)
        
        self.btn_treinar = ctk.CTkButton(painel_controle, text="Iniciar Treinamento",
                                       command=self._iniciar_treinamento_thread,
                                       fg_color="#2e7d32", hover_color="#1b5e20")
        self.btn_treinar.pack(side="right", padx=20, pady=10)
        
        self.lbl_status_treino = ctk.CTkLabel(painel_controle, text="Status: Não treinado")
        self.lbl_status_treino.pack(side="right", padx=20)
        
        # Barra de Progresso
        self.progress_bar = ctk.CTkProgressBar(frame)
        self.progress_bar.grid(row=2, column=0, sticky="ew", padx=20, pady=10)
        self.progress_bar.set(0)
        
        # Área do Gráfico de Loss
        self.frame_grafico_loss = ctk.CTkFrame(frame)
        self.frame_grafico_loss.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

    def _setup_aba_visualizacoes(self):
        frame = self.frames["visualizacoes"]
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        # Controles
        painel_vis = ctk.CTkFrame(frame)
        painel_vis.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        opcoes = ["Comparação 2D", "Erro Absoluto", "Superfície 3D", "Gregas"]
        self.combo_vis = ctk.CTkComboBox(painel_vis, values=opcoes, 
                                       command=self._atualizar_visualizacao)
        self.combo_vis.pack(side="left", padx=20, pady=10)
        self.combo_vis.set("Comparação 2D")
        
        ctk.CTkButton(painel_vis, text="Atualizar Gráfico", 
                     command=lambda: self._atualizar_visualizacao(self.combo_vis.get())).pack(side="left")
        
        # Área do Gráfico
        self.frame_grafico_vis = ctk.CTkFrame(frame)
        self.frame_grafico_vis.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

    # --- Lógica de Negócio ---

    def _atualizar_aba_modelo(self):
        caminho_img = os.path.join(os.path.dirname(__file__), 'equacoes', 'painel_completo.png')
        if os.path.exists(caminho_img):
            img = Image.open(caminho_img)
            # Ajusta tamanho mantendo proporção
            largura_display = 1000
            ratio = largura_display / img.width
            altura_display = int(img.height * ratio)
            
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, 
                                size=(largura_display, altura_display))
            self.lbl_img_equacoes.configure(image=img_tk, text="")

    def _calcular_black_scholes(self):
        try:
            params = {
                'preco_ativo': float(self.entradas_calc["Preço do Ativo (S)"].get()),
                'preco_strike': float(self.entradas_calc["Preço Strike (K)"].get()),
                'tempo_maturidade': float(self.entradas_calc["Tempo Maturidade (T anos)"].get()),
                'taxa_juros': float(self.entradas_calc["Taxa de Juros (r)"].get()),
                'volatilidade': float(self.entradas_calc["Volatilidade (σ)"].get())
            }
            
            self.modelo_bs.configurar_parametros(**params)
            resumo = self.modelo_bs.obter_resumo()
            
            self.lbl_resultado_call.configure(text=f"Call: R$ {resumo['precos']['call']:.4f}")
            self.lbl_resultado_put.configure(text=f"Put: R$ {resumo['precos']['put']:.4f}")
            
            for grega, valor in resumo['gregas_call'].items():
                nome_formatado = grega.capitalize()
                self.lbls_gregas[nome_formatado].configure(text=f"{nome_formatado}: {valor:.6f}")
                
        except ValueError as e:
            messagebox.showerror("Erro de Entrada", str(e))

    def _iniciar_treinamento_thread(self):
        if self.treinamento_em_andamento:
            return
            
        try:
            # Pega parâmetros da aba calculadora
            params = {
                'K': float(self.entradas_calc["Preço Strike (K)"].get()),
                'T': float(self.entradas_calc["Tempo Maturidade (T anos)"].get()),
                'r': float(self.entradas_calc["Taxa de Juros (r)"].get()),
                'sigma': float(self.entradas_calc["Volatilidade (σ)"].get())
            }
        except ValueError:
            messagebox.showerror("Erro", "Configure os parâmetros na aba Calculadora primeiro.")
            return

        self.treinamento_em_andamento = True
        self.btn_treinar.configure(state="disabled", text="Treinando...")
        self.progress_bar.set(0)
        
        # Cria nova instância da PINN
        self.pinn = RedeNeuralPINN(camadas=[2, 50, 50, 50, 1])
        
        thread = threading.Thread(target=self._executar_treino, args=(params,))
        thread.start()

    def _executar_treino(self, params):
        def callback(progresso, msg):
            self.after(0, lambda: self.progress_bar.set(progresso))
            self.after(0, lambda: self.lbl_status_treino.configure(text=msg))

        historico = self.pinn.treinar(epocas=1000, callback_progresso=callback, **params)
        
        self.after(0, lambda: self._finalizar_treino(historico))

    def _finalizar_treino(self, historico):
        self.treinamento_em_andamento = False
        self.pinn_treinada = True
        self.btn_treinar.configure(state="normal", text="Iniciar Treinamento")
        self.lbl_status_treino.configure(text="Treinamento Concluído!")
        
        # Plota histórico
        fig = self.visualizador.plotar_historico_treinamento(historico)
        self._exibir_grafico(fig, self.frame_grafico_loss)
        
        messagebox.showinfo("Sucesso", "Rede Neural treinada com sucesso!")

    def _atualizar_visualizacao(self, tipo_vis):
        if not self.pinn_treinada:
            ctk.CTkLabel(self.frame_grafico_vis, 
                        text="Treine a rede neural primeiro para ver as visualizações.").pack(pady=20)
            return
            
        # Limpa frame anterior
        for widget in self.frame_grafico_vis.winfo_children():
            widget.destroy()
            
        # Gera dados para plotagem
        S_range = np.linspace(50, 150, 100)
        t_fixo = 0.0 # Preço hoje
        
        # Previsões PINN
        precos_pinn = self.pinn.prever(S_range, t_fixo)
        
        # Preços Analíticos
        precos_analiticos = []
        for s in S_range:
            precos_analiticos.append(self.modelo_bs.calcular_superficie_precos([s], [self.pinn.T])[0][0])
            
        fig = None
        if tipo_vis == "Comparação 2D":
            fig = self.visualizador.plotar_comparacao_2d(S_range, precos_analiticos, precos_pinn, self.pinn.K)
        elif tipo_vis == "Erro Absoluto":
            erro = np.abs(np.array(precos_analiticos) - np.array(precos_pinn))
            fig = self.visualizador.plotar_erro_absoluto(S_range, erro)
        elif tipo_vis == "Superfície 3D":
            S_mesh = np.linspace(50, 150, 30)
            T_mesh = np.linspace(0, self.pinn.T, 30)
            S_grid, T_grid = np.meshgrid(S_mesh, T_mesh)
            
            # Previsão em lote
            S_flat = S_grid.flatten()
            T_flat = T_grid.flatten()
            V_pred = self.pinn.prever(S_flat, T_flat).reshape(S_grid.shape)
            
            fig = self.visualizador.plotar_superficie_3d(S_grid, T_grid, V_pred)
        elif tipo_vis == "Gregas":
            # Calcula gregas analíticas para comparação
            gregas = {'delta': [], 'gamma': [], 'vega': [], 'theta': []}
            for s in S_range:
                self.modelo_bs.preco_ativo = s
                g = self.modelo_bs.calcular_todas_gregas()
                for k in gregas: gregas[k].append(g[k])
            fig = self.visualizador.plotar_gregas(S_range, gregas)
            
        if fig:
            self._exibir_grafico(fig, self.frame_grafico_vis)

    def _exibir_grafico(self, fig, frame_container):
        # Limpa container
        for widget in frame_container.winfo_children():
            widget.destroy()
            
        canvas = FigureCanvasTkAgg(fig, master=frame_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Navegação ---
    def _ocultar_todos_frames(self):
        for frame in self.frames.values():
            frame.pack_forget()

    def _mostrar_modelo(self):
        self._ocultar_todos_frames()
        self.frames["modelo"].pack(fill="both", expand=True)

    def _mostrar_calculadora(self):
        self._ocultar_todos_frames()
        self.frames["calculadora"].pack(fill="both", expand=True)

    def _mostrar_pinn(self):
        self._ocultar_todos_frames()
        self.frames["pinn"].pack(fill="both", expand=True)

    def _mostrar_visualizacoes(self):
        self._ocultar_todos_frames()
        self.frames["visualizacoes"].pack(fill="both", expand=True)


if __name__ == "__main__":
    app = CalculadoraPINNApp()
    app.mainloop()
