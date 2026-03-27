import numpy as np
from numba import njit, prange
from itertools import combinations


# 3x3 desen 9 kernel kombinasyonu 84
INDICES = np.array(list(combinations(range(9), 3)), dtype=np.int32)

# filtrenin biaslarını altın orandan yüzde veriyoruz ondan sonra bias seçimi
def _quantiles(n):
    phi = (np.sqrt(5.0) + 1.0) / 2.0
    return np.array([(i * phi) % 1 for i in range(1, n + 1)],
                    dtype=np.float32)

def fit_dilations_2d(H, W, num_features, max_dilations=32):
    #84 farklı kombinasyon
    num_kernels = 84
    #eğer 100000 özellik istiyorsak yaklaşık 119 farklı dilation gerekir
    per_kernel = num_features // num_kernels
    true_dil = min(per_kernel, max_dilations)

    #filtre görüntü dışına taşmamması için
    max_d = min(H, W)
    max_exp = np.log2((max_d - 1) / 2)

    #mantık logfaritmik olarak genişleme miktarlarını arttırmak sebebp? uzaklaştıkça önem azalır
    base = np.logspace(0, max_exp, true_dil, base=2).astype(np.int32)
    base = np.unique(base)

    #filtre uzatmaları
    dilations = []
    for d in base:
        dilations.append((d, d))
        dilations.append((d, 1))
        dilations.append((1, d))

    return np.array(dilations, dtype=np.int32)


@njit(fastmath=True)
def _ppv_2d(img, dy, dx, i0, i1, i2, bias):

    H, W = img.shape
    pad_y, pad_x = dy, dx

    pos = 0
    total = 0

    #gezinti
    for y in range(pad_y, H - pad_y):
        for x in range(pad_x, W - pad_x):

            v0 = img[y - dy, x - dx]
            v1 = img[y - dy, x]
            v2 = img[y - dy, x + dx]

            v3 = img[y, x - dx]

            #merkez piksel
            v4 = img[y, x]
            v5 = img[y, x + dx]

            v6 = img[y + dy, x - dx]
            v7 = img[y + dy, x]
            v8 = img[y + dy, x + dx]

            vals = (v0, v1, v2, v3, v4, v5, v6, v7, v8)

            #evrişim işlemi eğer seçilen nokta merkez pikselden farklıysa s büyük olur
            s = -v4 + 3.0 * (vals[i0] + vals[i1] + vals[i2])
            
            #ppv işlemi ile bias kontrolü
            if s > bias:
                pos += 1
            total += 1

    return pos / total


@njit(fastmath=True)
def fit_biases_2d(X, dilations, quantiles):

    N = X.shape[0]
    num_kernels = 84
    num_dil = len(dilations)

    biases = np.empty(num_kernels * num_dil, dtype=np.float32)
    k = 0

    for d in range(num_dil):
        dy, dx = dilations[d]
        img = X[np.random.randint(N)]

        H, W = img.shape

        for kernel in range(num_kernels):
            i0, i1, i2 = INDICES[kernel]

            buf = []
            #ikişer ikişer atla
            for y in range(dy, H - dy, 2):
                for x in range(dx, W - dx, 2):

                    v0 = img[y - dy, x - dx]
                    v1 = img[y - dy, x]
                    v2 = img[y - dy, x + dx]
                    v3 = img[y, x - dx]
                    v4 = img[y, x]
                    v5 = img[y, x + dx]
                    v6 = img[y + dy, x - dx]
                    v7 = img[y + dy, x]
                    v8 = img[y + dy, x + dx]

                    vals = (v0, v1, v2, v3, v4, v5, v6, v7, v8)
                    s = -v4 + 3.0 * (vals[i0] + vals[i1] + vals[i2])
                    buf.append(s)

            biases[k] = np.quantile(np.array(buf), quantiles[k])
            k += 1

    return biases


@njit(parallel=True, fastmath=True)
def transform_2d(X, dilations, biases):

    N, H, W = X.shape
    num_kernels = 84
    num_dil = len(dilations)

    features = np.empty((N, num_kernels * num_dil), dtype=np.float32)

    for n in prange(N):
        img = X[n]
        k = 0

        for d in range(num_dil):
            dy, dx = dilations[d]

            for kernel in range(num_kernels):
                i0, i1, i2 = INDICES[kernel]
                features[n, k] = _ppv_2d(
                    img, dy, dx,
                    i0, i1, i2,
                    biases[k]
                )
                k += 1

    return features

# =================================================
# MAIN FIT FUNCTION
# =================================================
def fit_2d(X, num_features=10_000):

    H, W = X.shape[1], X.shape[2]

    dilations = fit_dilations_2d(H, W, num_features)
    total = 84 * len(dilations)

    quantiles = _quantiles(total)
    biases = fit_biases_2d(X, dilations, quantiles)

    return dilations, biases
