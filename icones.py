"""
Gerador de ícones e assets visuais para a Calculadora PINN + Black-Scholes
"""
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def criar_gradiente(largura, altura, cor1, cor2, direcao='vertical'):
    """Cria um gradiente entre duas cores"""
    imagem = Image.new('RGB', (largura, altura))
    desenho = ImageDraw.Draw(imagem)
    
    for i in range(altura if direcao == 'vertical' else largura):
        proporcao = i / (altura if direcao == 'vertical' else largura)
        r = int(cor1[0] * (1 - proporcao) + cor2[0] * proporcao)
        g = int(cor1[1] * (1 - proporcao) + cor2[1] * proporcao)
        b = int(cor1[2] * (1 - proporcao) + cor2[2] * proporcao)
        
        if direcao == 'vertical':
            desenho.line([(0, i), (largura, i)], fill=(r, g, b))
        else:
            desenho.line([(i, 0), (i, altura)], fill=(r, g, b))
    
    return imagem


def criar_icone_app():
    """Cria ícone principal da aplicação"""
    tamanho = 256
    imagem = Image.new('RGBA', (tamanho, tamanho), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(imagem)
    
    # Fundo com gradiente
    gradiente = criar_gradiente(tamanho, tamanho, (31, 83, 141), (139, 92, 246))
    imagem.paste(gradiente, (0, 0))
    
    # Círculo central
    margem = 20
    desenho.ellipse([margem, margem, tamanho-margem, tamanho-margem], 
                    fill=(255, 255, 255, 30), 
                    outline=(255, 255, 255, 200), 
                    width=4)
    
    # Desenha "PINN" estilizado (representação de rede neural)
    centro_x, centro_y = tamanho // 2, tamanho // 2
    
    # Camadas de neurônios
    raio_neuronio = 8
    espacamento = 50
    
    # Camada entrada (3 neurônios)
    for i in range(3):
        y = centro_y - espacamento + i * espacamento
        desenho.ellipse([40-raio_neuronio, y-raio_neuronio, 
                        40+raio_neuronio, y+raio_neuronio],
                       fill=(100, 200, 255), outline=(255, 255, 255))
    
    # Camada oculta (4 neurônios)
    for i in range(4):
        y = centro_y - 1.5*espacamento + i * espacamento
        desenho.ellipse([centro_x-raio_neuronio, y-raio_neuronio,
                        centro_x+raio_neuronio, y+raio_neuronio],
                       fill=(150, 100, 255), outline=(255, 255, 255))
        
        # Conexões
        for j in range(3):
            y_entrada = centro_y - espacamento + j * espacamento
            desenho.line([(40, y_entrada), (centro_x, y)], 
                        fill=(255, 255, 255, 100), width=1)
    
    # Camada saída (1 neurônio)
    desenho.ellipse([tamanho-40-raio_neuronio, centro_y-raio_neuronio,
                    tamanho-40+raio_neuronio, centro_y+raio_neuronio],
                   fill=(255, 150, 100), outline=(255, 255, 255))
    
    # Conexões para saída
    for i in range(4):
        y = centro_y - 1.5*espacamento + i * espacamento
        desenho.line([(centro_x, y), (tamanho-40, centro_y)],
                    fill=(255, 255, 255, 100), width=1)
    
    return imagem


def criar_icone_calcular():
    """Cria ícone para botão calcular"""
    tamanho = 64
    imagem = Image.new('RGBA', (tamanho, tamanho), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(imagem)
    
    # Círculo de fundo
    desenho.ellipse([2, 2, tamanho-2, tamanho-2],
                    fill=(31, 83, 141), outline=(100, 200, 255), width=2)
    
    # Símbolo de calculadora/gráfico
    margem = 16
    largura_barra = 6
    
    # Barras de gráfico
    alturas = [0.3, 0.6, 0.4, 0.7]
    for i, altura in enumerate(alturas):
        x = margem + i * (largura_barra + 4)
        y = tamanho - margem - int((tamanho - 2*margem) * altura)
        desenho.rectangle([x, y, x+largura_barra, tamanho-margem],
                         fill=(100, 200, 255))
    
    return imagem


def criar_icone_treinar():
    """Cria ícone para botão treinar rede neural"""
    tamanho = 64
    imagem = Image.new('RGBA', (tamanho, tamanho), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(imagem)
    
    # Círculo de fundo
    desenho.ellipse([2, 2, tamanho-2, tamanho-2],
                    fill=(139, 92, 246), outline=(200, 150, 255), width=2)
    
    # Símbolo de rede neural simplificado
    centro = tamanho // 2
    raio = 4
    
    # Neurônios
    posicoes = [
        (20, centro-10), (20, centro+10),  # Entrada
        (centro, centro-10), (centro, centro+10),  # Oculta
        (tamanho-20, centro)  # Saída
    ]
    
    # Desenha conexões
    for i in range(2):
        for j in range(2, 4):
            desenho.line([posicoes[i], posicoes[j]], fill=(255, 255, 255, 150), width=2)
    for i in range(2, 4):
        desenho.line([posicoes[i], posicoes[4]], fill=(255, 255, 255, 150), width=2)
    
    # Desenha neurônios
    for pos in posicoes:
        desenho.ellipse([pos[0]-raio, pos[1]-raio, pos[0]+raio, pos[1]+raio],
                       fill=(255, 255, 255))
    
    return imagem


def criar_icone_grafico():
    """Cria ícone para visualizações"""
    tamanho = 64
    imagem = Image.new('RGBA', (tamanho, tamanho), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(imagem)
    
    # Círculo de fundo
    desenho.ellipse([2, 2, tamanho-2, tamanho-2],
                    fill=(46, 125, 50), outline=(100, 255, 150), width=2)
    
    # Desenha curva (simulando gráfico)
    pontos = []
    for i in range(15):
        x = 12 + i * 3
        y = tamanho//2 + int(15 * np.sin(i * 0.5))
        pontos.append((x, y))
    
    desenho.line(pontos, fill=(100, 255, 150), width=3, joint='curve')
    
    # Eixos
    desenho.line([(10, tamanho-10), (tamanho-10, tamanho-10)], 
                fill=(255, 255, 255), width=2)
    desenho.line([(10, 10), (10, tamanho-10)], 
                fill=(255, 255, 255), width=2)
    
    return imagem


def criar_todos_icones():
    """Cria todos os ícones do projeto"""
    diretorio = os.path.dirname(os.path.abspath(__file__))
    dir_icones = os.path.join(diretorio, 'icones')
    
    # Cria diretório se não existir
    os.makedirs(dir_icones, exist_ok=True)
    
    print("🎨 Gerando ícones...")
    
    # Ícone principal
    icone_app = criar_icone_app()
    icone_app.save(os.path.join(dir_icones, 'app_icon.png'))
    icone_app.resize((128, 128)).save(os.path.join(dir_icones, 'app_icon_128.png'))
    icone_app.resize((64, 64)).save(os.path.join(dir_icones, 'app_icon_64.png'))
    icone_app.resize((32, 32)).save(os.path.join(dir_icones, 'app_icon_32.png'))
    print("  ✓ Ícone principal criado")
    
    # Ícones de botões
    criar_icone_calcular().save(os.path.join(dir_icones, 'calcular.png'))
    print("  ✓ Ícone calcular criado")
    
    criar_icone_treinar().save(os.path.join(dir_icones, 'treinar.png'))
    print("  ✓ Ícone treinar criado")
    
    criar_icone_grafico().save(os.path.join(dir_icones, 'grafico.png'))
    print("  ✓ Ícone gráfico criado")
    
    # Cria imagem de fundo com gradiente
    fundo = criar_gradiente(1400, 900, (20, 30, 48), (31, 83, 141))
    fundo.save(os.path.join(dir_icones, 'fundo_gradiente.png'))
    print("  ✓ Fundo gradiente criado")
    
    print(f"\n✅ Todos os ícones foram salvos em: {dir_icones}")
    return dir_icones


if __name__ == "__main__":
    criar_todos_icones()
