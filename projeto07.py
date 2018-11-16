import cv2
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec2
from pywt import waverec2

def plot_img(img, l = 256):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap = 'gray', vmin = 0, vmax = l - 1)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def blockproc(image, m, n, fun):    
    #### Padding
    pad_x = image.shape[0] % m
    pad_y = image.shape[0] % n
    if pad_x != 0 or pad_y != 0:
        image = np.pad(image, ((0, pad_x), (0, pad_y)), mode = 'constant', constant_values = 0)
    ####
    for i in range(0, image.shape[0], m):
        for j in range(0, image.shape[1], n):
            image[i:i+m, j:j+n] = fun(image[i:i+m, j:j+n])
    #### Slicing
    if pad_x > 0:
        image = image[:pad_x * -1, :]
    if pad_y > 0:
        image = image[:pad_y * -1, :]
    ####
    return image
        
def quantizar(T, Q):
    return (T // Q) * Q
    
def perc_nonzero(transformada):
    return np.count_nonzero(transformada) / transformada.size * 100

def plot_Q_nonzero(list_Q, list_nonzero, name_transf):
    fig, ax = plt.subplots()
    ax.plot(list_Q, list_nonzero)
    plt.xlim(list_Q[0], 0)  # decreasing time
    ax.set(xlabel='Quantização', ylabel='%nonZ',
           title='Porcentagem de coeficientes não nulos por Quantização\n' + name_transf)
    ax.grid()
    #fig.savefig("name.png")
    plt.show()
    
def plot_nonzero_psnr(list_nonzero, list_psnr, name_transf):
    fig, ax = plt.subplots()
    ax.plot(list_nonzero, list_psnr)
    plt.xlim(list_nonzero[0], list_nonzero[-1])  # decreasing time
    ax.set(xlabel='%nonZ', ylabel='PSNR',
           title='PSNR por Porcentagem de coeficientes não nulos\n' + name_transf)
    ax.grid()
    #fig.savefig("name.png")
    plt.show()

def curva_quantizacao(transformada, name_transf):
    list_nonzero = []
    list_Q = range(50, 0, -1)
    for Q in list_Q:
        list_nonzero.append(perc_nonzero(quantizar(transformada, Q)))
    plot_Q_nonzero(list_Q, list_nonzero, name_transf)
    return list_Q, list_nonzero

def curva_reconstrucao(image, transformada, calcula_inversa, name_transf):
    list_Q, list_nonzero = curva_quantizacao(transformada, name_transf)
    list_psnr = []
    for Q in list_Q:
        transf_quant = quantizar(transformada, Q)
        image_quant = calcula_inversa(transf_quant)
        list_psnr.append(psnr(image, image_quant))
    plot_nonzero_psnr(list_nonzero, list_psnr, name_transf)

def quantizar_wavelets(transformada, Q):
    T = transformada.copy()
    T[0] = quantizar(T[0], Q)
    for lev in range(1,len(T)):
        T[lev] = list(T[lev])
        for k in range(3):
            T[lev][k] = quantizar(T[lev][k], Q)
    return T

def perc_nonzero_wavelets(transformada):
    n_nonzero = 0
    n_total = 0
    n_nonzero += np.count_nonzero(transformada[0])
    n_total += transformada[0].size
    for lev in range(1,len(transformada)):
        for k in range(3):
            n_nonzero += np.count_nonzero(transformada[lev][k])
            n_total += transformada[lev][k].size
    return n_nonzero / n_total * 100

def curva_quantizacao_wavelets(transformada, name_transf):
    list_nonzero = []
    list_Q = range(50, 0, -1)
    for Q in list_Q:
        list_nonzero.append(perc_nonzero_wavelets(quantizar_wavelets(transformada, Q)))
    plot_Q_nonzero(list_Q, list_nonzero, name_transf)
    return list_Q, list_nonzero   

def curva_reconstrucao_wavelets(image, transformada, base, name_transf):
    list_Q, list_nonzero = curva_quantizacao_wavelets(transformada, name_transf)
    list_psnr = []
    for Q in list_Q:
        transf_quant = quantizar_wavelets(transformada, Q)
        image_quant = waverec2(transf_quant, base)
        list_psnr.append(psnr(image, image_quant))
    plot_nonzero_psnr(list_nonzero, list_psnr, name_transf)
    
    
def psnr(image, image_quant):
    return 20 * np.log10(255 / np.sqrt(mse(image, image_quant)))

def mse(image, image_quant):
    soma = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            soma += (image[i,j] - image_quant[i,j])**2
    soma = soma / image.size
    return soma

def calc_idct_block(transformada):
    return blockproc(transformada, 8, 8, cv2.idct)

def calc_waverec_haar(transf_image):
    transformada = mount_wavelet_dict(transf_image, 5)
    return waverec2(transformada, 'haar')

def mount_wavelet_image(wavelets_transform): #monta a TW como uma imagem
    wavelets_image = np.zeros(np.multiply(wavelets_transform[-1][0].shape, 2))
    [m, n] = wavelets_transform[0].shape
    wavelets_image[:m, :n] = wavelets_transform[0] #adding cAn
    for lev in wavelets_transform[1:]:
        wavelets_image[:m, n:n+n] = lev[0] #adding cHk
        wavelets_image[m:m+m, :n] = lev[1] #adding cVk
        wavelets_image[m:m+m, n:n+n] = lev[2] #adding cDk
        m = 2 * m 
        n = 2 * n
    return wavelets_image

def mount_wavelet_dict(wavelets_image, levels): #remonta a imagem TW como dicionario 
    wavelets_transform = dict()
    [m, n] = wavelets_image.shape
    for k in range(levels, 0, -1):
        m //= 2
        n //= 2
        wavelets_transform[k] = [wavelets_image[:m, n:n+n], wavelets_image[m:m+m, :n], wavelets_image[m:m+m, n:n+n]]     
    wavelets_transform[0] = wavelets_image[:m, :n]
    return wavelets_transform   
    
def go(filename):
    image = cv2.imread(dir_images + '/' + filename, 0)
    plot_img(image)
    
    #Aplicar DCT a imagem globalmente
    dct_image = image/255.0  # float conversion/scale
    dct_image = cv2.dct(dct_image)           # the dct
    dct_image = dct_image*255.0    # convert back
    plot_img(dct_image)
    
    #Aplicar a DCT a imagem em blocos 8x8
    dct_block = image/255.0  # float conversion/scale
    dct_block = blockproc(dct_block, 8, 8, cv2.dct)           # the dct
    dct_block = dct_block*255.0    # convert bacdef perc_nonzero(transf):
    plot_img(dct_block)
    
    
    #Aplicar Wavelets Discreta a imagem globalmente usando Haar
    wavelets_haar = wavedec2(image, 'haar', level=5)
    plot_img(mount_wavelet_image(wavelets_haar))
    
    #Aplicar Wavelets Discreta a imagem globalmente usando db4
    wavelets_db4 = wavedec2(image, 'db4', level=5)
       
    
    curva_reconstrucao(image, dct_image, cv2.idct, 'DCT Global')
    curva_reconstrucao(image, dct_block, calc_idct_block, 'DCT por blocos de 8x8')
    curva_reconstrucao_wavelets(image, wavelets_haar, 'haar', 'Wavelets Haar')
    curva_reconstrucao_wavelets(image, wavelets_db4, 'db4', 'Wavelets DB4')

filenames = ['Lena.bmp', 'WonderWoman.png', 'Lagertha.png']
dir_images = 'images'

for filename in filenames:
    go(filename)