import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2


# Define a divergence gradient operator
def div(u1, u2):
    if u1.shape != u2.shape:
        print('Arrays have different shapes')
    else:
        N = u1.shape[0]
        M = u1.shape[1]
        div1 = np.zeros((N, M))
        div2 = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                if i == 1:
                    div1[i, j] = u1[i, j]
                else:
                    if 1 < i < N:
                        div1[i, j] = u1[i, j] - u1[i-1, j]
                    else:
                        div1[i, j] = -u1[i-1, j]
                if j == 1:
                    div2[i, j] = u2[i, j]
                else:
                    if 1 < j < M:
                        div2[i, j] = u2[i, j] - u2[i, j-1]
                    else:
                        div2[i, j] = -u2[i, j-1]
        return div1+div2


# Defining a discrete gradient operator
def grad(u):
    N = u.shape[0]
    M = u.shape[1]
    grad1 = np.zeros((N, M))
    grad2 = np.zeros((N, M))
    for i in range(N-1):
        for j in range(M-1):
            if i < N:
                grad1[i, j] = u[i+1, j] - u[i, j]
            else:
                grad1[i, j] = 0
            if j < M:
                grad2[i, j] = u[i, j+1] - u[i, j]
            else:
                grad2[i, j] = 0
    return grad1, grad2


# Tests
img = cv2.imread('data/IMG_0410.PNG', 0)[38:731, 8:681]
#img = imread('data/IMG_0410.PNG')[38:731, 8:681]
u = np.random.normal(0, 1, (40, 40)) + np.random.gamma(2, 1, (40, 40))
v = np.random.normal(0, 1, (643, 673))
print(type(img))
print(img.shape)
print(type(u))
print(u.shape)

plt.figure()
plt.subplot(221)
plt.imshow(img)
plt.title('Original image (1-color scaled)')
plt.subplot(222)
plt.imshow(div(img, v))
plt.title('Divergence (div(image, gaussian_noise) here)')
plt.subplot(223)
plt.imshow(grad(img)[1])
plt.title('Horizontal gradient')
plt.subplot(224)
plt.imshow(grad(img)[0])
plt.title('Vertical gradient')
plt.tight_layout()

plt.show()
