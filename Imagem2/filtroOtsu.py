import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem_path = "Imagem2/Esfregaco-de-sangue.png"  # Substitua pelo caminho real da sua imagem
imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

# Aplicar o método de Otsu
_, imagem_segmentada = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Exibir as imagens
plt.subplot(121), plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(imagem_segmentada, cmap='gray')
plt.title('Imagem Segmentada - Otsu'), plt.xticks([]), plt.yticks([])

plt.show()

