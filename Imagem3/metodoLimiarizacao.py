import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem_path = "Imagem3/imgem3.jpg" 
imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

# Aplicar limiarização adaptativa
imagem_limiarizada = cv2.adaptiveThreshold(imagem, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Exibir as imagens
plt.subplot(121), plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(imagem_limiarizada, cmap='gray')
plt.title('Imagem Limiarizada Adaptativa'), plt.xticks([]), plt.yticks([])

plt.show()