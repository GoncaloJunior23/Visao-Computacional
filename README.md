Projeto elaborado por Gonçalo Júnior e Juliana Braga para matéria de Inteligência Artificial.


# Visão Computacional para Monitoramento de Carros

Este projeto utiliza técnicas de visão computacional para identificar e contar o número de carros em um vídeo. Ele processa cada quadro do vídeo, destaca os veículos detectados e exibe o número total de carros em tempo real. O vídeo processado é salvo automaticamente para análise posterior.

## Funcionalidades

- Detecção de veículos em vídeos usando um modelo pré-treinado de TensorFlow.
- Contagem do número de carros por quadro.
- Gravação do vídeo processado com as detecções destacadas.
- Opções de pausa e retomada durante a execução.

## Requisitos

- Python 3.8 ou superior
- OpenCV 4.5 ou superior
- NumPy

## Configuração

1. Certifique-se de que os seguintes arquivos estejam disponíveis:
   - **Arquivo de vídeo**: `rastreio-carros/video casa.mp4`
   - **Modelo pré-treinado**: `rastreio-carros/frozen_inference_graph.pb`
   - **Arquivo de configuração do modelo**: `rastreio-carros/ssd_mobilenet_v2_coco.pbtxt`

2. Instale as dependências do Python:

```bash
pip install opencv-python opencv-python-headless numpy

## Como Executar

* Certifique-se de que os caminhos dos arquivos estão configurados corretamente no código:

ARQUIVO_VIDEO = 'rastreio-carros/walking.mp4'
ARQUIVO_MODELO = 'rastreio-carros/frozen_inference_graph.pb'
ARQUIVO_CFG = 'rastreio-carros/ssd_mobilenet_v2_coco.pbtxt'

## Execute o script:

python monitoramento_carros.py

## Durante a execução:

Pressione P para pausar ou retomar o vídeo.
Pressione Q para sair.

* O vídeo processado será salvo automaticamente como saida.avi no mesmo diretório.

## Estrutura do Código

Função processa_frame

* A função processa_frame é responsável por detectar veículos no vídeo e destacá-los com caixas delimitadoras:

Converte os quadros para um formato adequado para o modelo pré-treinado.
Aplica Supressão Não Máxima (NMS) para remover caixas sobrepostas.
Desenha as caixas nos veículos detectados.
Conta e exibe o número total de veículos no quadro.

## Gravação do Vídeo
O vídeo processado com as detecções é salvo automaticamente em saida.avi usando o codec XVID.

## Exemplo de Saída
Durante a execução, será exibida uma janela com o vídeo processado. Os veículos detectados serão destacados com caixas verdes, e o número total será exibido no canto superior esquerdo.

O arquivo de saída será salvo automaticamente no formato .avi.