import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans


def loadData(path):
    f = open(path, 'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    f.close()
    return np.mat(data), m, n


if __name__ == '__main__':
    imgData, row, col = loadData('C:\images.jpg')
    km = KMeans(n_clusters=3)
    label = km.fit_predict(imgData).reshape(row, col)
    pic_new = image.new('L', (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i, j), label[i][j])
    pic_new.save('C:\iresult.jpg','JPEG')
