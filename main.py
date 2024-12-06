import cv2
import numpy as np

# Caminhos dos arquivos
ARQUIVO_VIDEO = 'rastreio-carros/video casa.mp4'
ARQUIVO_MODELO = 'rastreio-carros/frozen_inference_graph.pb'
ARQUIVO_CFG = 'rastreio-carros/ssd_mobilenet_v2_coco.pbtxt'

def carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG):
    '''
    Carrega o modelo de deep learning do TensorFlow para detecção de objetos.
    ARQUIVO_MODELO: Caminho para o arquivo .pb contendo os pesos do modelo.
    ARQUIVO_CFG: Caminho para o arquivo .pbtxt contendo a configuração do modelo.
    Retorna o modelo carregado.
    '''
    try:
        modelo = cv2.dnn.readNetFromTensorflow(ARQUIVO_MODELO, ARQUIVO_CFG)
    except cv2.error as erro:
        print(f"Erro ao carregar o modelo: {erro}")
        exit()
    return modelo

def aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf, limiar_supr):
    '''
    Aplica a Supressão Não Máxima para reduzir o número de caixas delimitadoras sobrepostas.
    caixas: Lista de caixas delimitadoras.
    confiancas: Lista de confianças de cada caixa.
    limiar_conf: Limiar de confiança para considerar detecções.
    limiar_supr: Limiar de sobreposição para suprimir caixas redundantes.
    Retorna uma lista de caixas após aplicar a supressão.
    '''
    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar_conf, limiar_supr)
    return [caixas[i] for i in indices.flatten()] if len(indices) > 0 else []

def main():
    '''
    Função principal que executa o rastreio de veículos no vídeo.
    '''
    captura = cv2.VideoCapture(ARQUIVO_VIDEO)
    detector_carros = carregar_modelo(ARQUIVO_MODELO, ARQUIVO_CFG)
    pausado = False

    # Configurar o gravador de vídeo
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(captura.get(cv2.CAP_PROP_FPS))
    gravador = cv2.VideoWriter('saida.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (largura, altura))

    while True:
        if not pausado:
            ret, frame = captura.read()
            if not ret:
                break

            # Criação do blob a partir do frame e realização da detecção
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            detector_carros.setInput(blob)
            deteccoes = detector_carros.forward()

            caixas = []
            confiancas = []

            # Extração das caixas delimitadoras e confianças das detecções
            for i in range(deteccoes.shape[2]):
                confianca = deteccoes[0, 0, i, 2]
                if confianca > 0.5:
                    (altura_frame, largura_frame) = frame.shape[:2]
                    caixa = deteccoes[0, 0, i, 3:7] * np.array([largura_frame, altura_frame, largura_frame, altura_frame])
                    (inicioX, inicioY, fimX, fimY) = caixa.astype("int")
                    caixas.append([inicioX, inicioY, fimX - inicioX, fimY - inicioY])
                    confiancas.append(float(confianca))

            # Aplicação da supressão não máxima para finalizar as caixas delimitadoras
            caixas_finais = aplicar_supressao_nao_maxima(caixas, confiancas, limiar_conf=0.5, limiar_supr=0.4)
            numero_veiculos = len(caixas_finais)

            # Desenho das caixas e exibição do número de veículos detectados
            for (inicioX, inicioY, largura, altura) in caixas_finais:
                cv2.rectangle(frame, (inicioX, inicioY), (inicioX + largura, inicioY + altura), (0, 255, 0), 2)
            cv2.putText(frame, f"Veículos: {numero_veiculos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Salvar o frame processado no vídeo
            gravador.write(frame)

        # Exibição do frame processado e controle de pausa/play
        cv2.imshow("Rastreio de Veículos", frame)
        
        tecla = cv2.waitKey(30) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    # Liberação dos recursos ao finalizar
    captura.release()
    gravador.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
