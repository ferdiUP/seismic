import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sp
from skimage.restoration import denoise_tv_chambolle


# Warning: this script contains a naïve implementation of the correlation between two images within given size windows,
# so the main function may be not perfectly correct. Also the complexity is quite huge so the algorithm is very slow
# on large images.
# However, we show the correlation plot computed with FFT.


# Defines the normalised correlation computed on a first image I1(t) and a second I2(t) = I1(t+dt), noted I1 & I2
# The correlation is taken for each window of size (2*M+1)*(2*N+1)
def normalised_correlation(I1, I2, M, N):
    v_ind = np.zeros((2*M+1, 2*N+1, 2))
    print(I1.shape)
    for i in range(M+1, I1.shape[0]-M+2):
        for j in range(N+1, I1.shape[1]-N+2):
            F1 = I1[M+1:I1.shape[0]-M+1, N+1:I1.shape[1]-N+1]
            print(F1.shape)
            C = np.zeros((I2.shape[0]-2*M+1, I2.shape[1]-2*N+1))
            print(I2.shape[0]-2*M)
            print(I2.shape[1]-2*N)
            for i2 in range(M+1, I2.shape[0]-M+2):
                for j2 in range(N+1, I2.shape[1]-N+2):
                    F2 = I2[M+1:I2.shape[0]-M+1, N+1:I2.shape[1]-N+1]
                    sum_n = 0
                    sum_d1 = 0
                    sum_d2 = 0
                    # Computes correlation between F1
                    for m in range(I2.shape[0]-2*M):
                        for n in range(I2.shape[1]-2*N):
                            sum_n += F1[m, n] * F2[m + i, n + j]
                            sum_d1 += (F1[m, n] - np.mean(F1)) ** 2
                            sum_d2 += (F2[m, n] - np.mean(F2)) ** 2
                    C[i2, j2] = sum_n/(sum_d1+sum_d2)
            # Takes the maximum correlation indices
            v_ind[i, j, 0] = np.argmax(np.linalg.norm(C))[0]
            v_ind[i, j, 1] = np.argamx(C)[1]
            # Prepares displacement vector field plot
            x, y = np.meshgrid(np.linspace(0, 2*M+1, 1), np.linspace(0, 2*N+1, 1))
            # Computes the displacement vector by subtracting indices from v_ind to indices from I1
            v = v_ind[:, :, 0]-x + v_ind[:, :, 1]-y
    return v_ind


# Defines the correlation using FFT
def fft_corr(a, b):
    image_product = np.fft.fft2(a) * np.fft.fft2(b).conj()
    return np.fft.fftshift(np.fft.ifft2(image_product)).real


if __name__ == "__main__":
    # Loads the data
    data1 = np.array([np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL94.txt', delimiter=','),
                      np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL01.txt', delimiter=','),
                      np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL04.txt', delimiter=','),
                      np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL06.txt', delimiter=',')])
    data2 = np.array([np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x94_2.txt', delimiter=','),
                      np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x01_2.txt', delimiter=','),
                      np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x04_2.txt', delimiter=','),
                      np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x06_2.txt', delimiter=',')])

    # Plots I(t) and I(t+dt)
    plt.figure()
    plt.subplot(121)
    plt.imshow(data1[1], cmap='seismic')
    plt.axis('auto')
    plt.title('2001 signal')

    plt.subplot(122)
    plt.imshow(data1[2], cmap='seismic')
    plt.axis('auto')
    plt.title('2004 signal')

    plt.figure()
    plt.imshow(data1[2]-data1[1], cmap='seismic')
    plt.axis('auto')
    plt.colorbar()
    plt.title('Difference $I(x, y, 2004) - I(x, y, 2001)$')

    # plt.figure()
    # plt.imshow(data1[1, 415:600, 85:250], cmap='seismic')

    # corr = normalised_correlation(data1[1, 415:600, 85:250], data1[2, 415:600, 85:250], 20, 5)
    # plt.figure()
    # plt.imshow(corr)

    # print(normalised_correlation(data1[1, 415:600, 85:250], data1[2, 415:600, 85:250], 200, 70))

    # 2D FFT-based correlation
    plt.figure()
    plt.imshow(fft_corr(data1[1], data1[2]))
    plt.title('Correlation between 2001 and 2004 signals')
    plt.axis('auto')
    plt.colorbar()

    # # Motion field estimation using phase correlation
    # plt.figure()
    # corr = phase_cross_correlation(data1[1], data1[2])
    # print(type(corr))
    # plt.imshow(corr)
    # plt.axis('auto')

    # 2D correlation on denoised images
    # Total variation denoising
    img1 = denoise_tv_chambolle(data1[1], weight=0.6, eps=0.00002, max_num_iter=10000,
                     multichannel=False, channel_axis=None)
    img2 = denoise_tv_chambolle(data1[2], weight=0.6, eps=0.00002, max_num_iter=10000,
                                multichannel=False, channel_axis=None)
    #
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1, cmap='seismic')
    plt.title('Denoised 2001 data')
    plt.axis('auto')
    plt.subplot(122)
    plt.imshow(img2, cmap='seismic')
    plt.title('Denoised 2004 data')
    plt.axis('auto')

    plt.figure()
    plt.imshow(fft_corr(img1, img2))
    plt.axis('auto')
    plt.title('Correlation between the 2 denoised images')
    plt.colorbar()

    # 1D FFT-base correlation on a x slice of the signal
    t = np.linspace(0, data1[1].shape[0], data1[1].shape[0])
    plt.figure()
    plt.subplot(311)
    plt.plot(data1[1][170, :])
    plt.title('2001 signal, slice $x=170$')
    plt.subplot(312)
    plt.plot(data1[2][170, :])
    plt.title('2004 signal, slice $x=170$')
    plt.subplot(313)
    plt.plot(sp.correlate(data1[1][170, :], data1[2][170, :]))
    plt.title('1D correlation between $I(170, x, 2001)$ and $I(170, x, 2004)$')
    plt.tight_layout()
    plt.show()
