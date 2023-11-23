import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem_path = "Imagem3/imgem3.jpg" 
imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

# Detecção de bordas 
bordas = cv2.Canny(imagem, 50, 150) 

# Exibir a imagem original e a imagem com bordas
plt.subplot(121), plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original em preto e branco'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(bordas, cmap='gray')
plt.title('Detecção de Bordas - Canny'), plt.xticks([]), plt.yticks([])

plt.show()
