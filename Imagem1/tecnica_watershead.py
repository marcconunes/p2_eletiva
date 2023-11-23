import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem_path = "Imagem1/041.png.webp"  
imagem = cv2.imread(imagem_path)
imagem_original = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Converter a imagem para escala de cinza
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização para separar o plano de fundo do primeiro plano
_, thresh = cv2.threshold(imagem_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Aplicar transformação morfológica (abertura) para remover pequenos ruídos
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Encontrar a sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Encontrar a sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Subtrair a sure foreground da sure background para obter a região desconhecida
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marcar as regiões conectadas na região desconhecida
_, markers = cv2.connectedComponents(sure_fg)

# Adicionar 1 aos marcadores de sure_fg para garantir que o plano de fundo seja 1, não 0
markers = markers + 1
markers[unknown == 255] = 0

# Aplicar Watershed
imagem_watershed = imagem_original.copy()
cv2.watershed(imagem, markers)
imagem_watershed[markers == -1] = [255, 0, 0]  # Pintar as bordas identificadas pelo Watershed de vermelho

# Exibir as imagens
plt.subplot(131), plt.imshow(imagem_original)
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(sure_bg, cmap='gray')
plt.title('Sure Background'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(imagem_watershed)
plt.title('Segmentação Watershed'), plt.xticks([]), plt.yticks([])

plt.show()
