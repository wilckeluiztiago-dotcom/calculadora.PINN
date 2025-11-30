# Calculadora PINN + Black-Scholes 🧠📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Uma aplicação sofisticada que combina **Redes Neurais Informadas pela Física (PINNs)** com o modelo clássico de **Black-Scholes** para precificação de opções financeiras. Desenvolvida com uma interface gráfica moderna e interativa.

## ✨ Funcionalidades

- **Interface Premium**: Design moderno com tema escuro, desenvolvido em `CustomTkinter`.
- **Modelo Híbrido**:
  - **Analítico**: Solução exata de Black-Scholes para validação.
  - **PINN**: Rede neural profunda que aprende a resolver a EDP de Black-Scholes sem supervisão direta de dados de preço, apenas usando a física do problema.
- **Visualizações Avançadas**:
  - Gráficos 3D interativos de superfícies de preço.
  - Mapas de calor e gráficos de erro.
  - Visualização em tempo real do treinamento da rede.
- **Equações Renderizadas**: Explicações matemáticas detalhadas com equações em LaTeX de alta qualidade.
- **Cálculo de Gregas**: Delta, Gamma, Vega, Theta e Rho.

## 🚀 Instalação

Recomendamos o uso de um ambiente virtual para gerenciar as dependências.

1. **Clone o repositório** (ou baixe os arquivos):
   ```bash
   # Navegue até a pasta do projeto
   cd "/home/luiztiagowilcke188/Área de trabalho/Projetos/CalculadoraPINN"
   ```

2. **Crie e ative um ambiente virtual**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   # .\venv\Scripts\activate  # Windows
   ```

3. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Como Usar

Execute a aplicação principal:

```bash
python3 calculadora_pinn_bs.py
```

### Navegação

1. **Modelo**: Entenda a teoria por trás da aplicação.
2. **Calculadora**: Insira os parâmetros (S, K, T, r, σ) e obtenha preços e gregas instantaneamente.
3. **PINN**: Treine a rede neural para aprender a precificar a opção configurada. Acompanhe a perda (loss) caindo em tempo real.
4. **Visualizações**: Compare os resultados da PINN com o modelo analítico em gráficos 2D e 3D.

## 🧠 Teoria: Physics-Informed Neural Networks

Diferente de redes neurais tradicionais que aprendem de pares (entrada, saída), uma PINN incorpora a Equação Diferencial Parcial (EDP) diretamente na função de perda:

$$ \mathcal{L} = \mathcal{L}_{Dados} + \mathcal{L}_{Física} $$

Para Black-Scholes, a "Física" é a própria EDP:

$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0 $$

A rede aprende a função $V(S,t)$ que minimiza o resíduo dessa equação, respeitando as condições de contorno e iniciais (payoff).

## 🛠️ Tecnologias

- **Python 3**: Linguagem base.
- **TensorFlow**: Construção e treinamento da PINN.
- **CustomTkinter**: Interface gráfica moderna.
- **Matplotlib**: Visualização de dados e renderização de LaTeX.
- **NumPy/SciPy**: Computação numérica e estatística.

## 📝 Licença

Este projeto está sob a licença MIT. Sinta-se livre para usar e modificar.

---
Desenvolvido com ❤️ por Antigravity
