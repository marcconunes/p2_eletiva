import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem_path = "Imagem1/041.png.webp"  
imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

# Aplicar o filtro Sobel para detecção de bordas
sobel_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)

# Exibir a imagem original e a magnitude do gradiente
plt.subplot(131), plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.show()
