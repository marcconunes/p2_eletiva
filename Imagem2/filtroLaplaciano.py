import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem_path = "Imagem2/Esfregaco-de-sangue.png" 
imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

# Aplicar o filtro Laplaciano
laplaciano = cv2.Laplacian(imagem, cv2.CV_64F)

# Calcular a imagem realçada (soma da imagem original com o resultado do filtro Laplaciano)
imagem_realcada = cv2.convertScaleAbs(imagem + laplaciano)

# Exibir as imagens
plt.subplot(131), plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(laplaciano, cmap='gray')
plt.title('Filtro Laplaciano'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(imagem_realcada, cmap='gray')
plt.title('Imagem Realçada'), plt.xticks([]), plt.yticks([])
plt.show()


