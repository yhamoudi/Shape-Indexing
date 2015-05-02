import sys
import re # regular expressions
import numpy as np
from scipy import ndimage, misc
from matplotlib import pyplot
from scipy.misc import imrotate
import random

class Image:
    def __init__(self, filename):
        f = open(filename, 'rb')
        line = f.readline().decode()
        while not line[0].isdigit():
            line = f.readline().decode()
        [width,height] = re.findall(r'\d+', line)
        maxval = re.findall(r'\d+', f.readline().decode())
        image = np.frombuffer(f.read(),
                              dtype='u1',
                              count=int(width)*int(height)
        ).reshape((int(height), int(width)))
        # boolean matrix : False if black, True if white (the picture itself is white) :
        self.image = np.vectorize(lambda x: x != 0)(image)

    def print(self):
        pyplot.imshow(self.image, pyplot.cm.gray)
        pyplot.show()

    def height(self):
        return self.image.shape[0]

    def width(self):
        return self.image.shape[1]

    def resize(self, max_edge_size):
        alpha = float(max_edge_size) / float(max(self.height(), self.width()))
        self.image = misc.imresize(self.image, alpha)
        self.image = np.vectorize(lambda x: x > 0.5)(self.image)

    def normalize(self,size):
        self.__reverse_colors()  # reverse colors if the picture itself was white
        self.crop()            # crop the image until all borders contain a white pixel
        self.resize(size)        # resize the image (largest side has size "size")
        self.__add_black_edges() # add a black border all around the image

    def rotate(self,angle): # rotate (and resize in order not to crop the initial image)
        exp = int(max(self.height(), self.width())/2)
        for i in range (0,exp):
            self.image = np.vstack((self.image,np.zeros(self.width())))
            self.image = np.vstack((np.zeros(self.width()),self.image))
        for i in range (0,exp):
            self.image = np.column_stack((self.image,np.zeros(self.height())))
            self.image = np.column_stack((np.zeros(self.height()),self.image))
        self.image = imrotate(self.image,angle)

    def noise(self,a): # Add a Kanungo noise of factor a
        dist = self.__distance()
        for i in range(0,dist.shape[0]):
          for j in range(0,dist.shape[1]):
            self.image[i][j] = random.random () < 1 - a**dist[i][j]

    ################ Private methods ##################

    def __neighbors(self,i,j): # All neighbors (a,b) of (i,j) that are not out of range
        x = [i]
        if i >= 1:
            x.append(i-1)
        if i <= self.image.shape[0] -2:
            x.append(i+1)
        y = [j]
        if j >= 1:
            y.append(j-1)
        if j <= self.image.shape[1] -2:
            y.append(j+1)
        res = []
        for a in x:
            for b in y:
                if a != i or b != j:
                    res.append([a,b])
        return res

    def __distance(self): # Distances of white pixels to the border
        dist = np.vectorize(lambda x: -1)(np.zeros(self.image.shape))
        still = True
        for i in range(0,dist.shape[0]):
            for j in range(0,dist.shape[1]):
                if not self.image[i][j]:
                  dist[i][j] = 0
        d = 1
        while still:
            still = False
            for i in range(0,dist.shape[0]):
                for j in range(0,dist.shape[1]):
                    if dist[i][j] == -1:
                        still = True
                    elif dist[i][j] == d-1:
                        for x in self.__neighbors(i,j):
                            if dist[x[0]][x[1]] == -1:
                                dist[x[0]][x[1]] = d
            d = d+1
        return dist

    def __reverse_colors(self):
        if self.image[0,:].sum() + self.image[self.height()-1,:].sum() + self.image[:,0].sum() + self.image[:, self.width()-1].sum() > self.height() + self.width():
            self.image = np.vectorize(lambda x: x == 0)(self.image)

    def __remove_first_line(self):
        if self.image[0,:].sum() == 0:
            self.image = self.image[1:, :]
            return self.__remove_first_line()

    def __remove_last_line(self):
        if self.image[self.height()-1,:].sum() == 0:
            self.image = self.image[0:self.height()-1, :]
            return self.__remove_last_line()

    def __remove_first_column(self):
        if self.image[:,0].sum() == 0:
            self.image = self.image[:, 1:]
            return self.__remove_first_column()

    def __remove_last_column(self):
        if self.image[:, self.width()-1].sum() == 0:
            self.image = self.image[:, 0:self.width()-1]
            return self.__remove_last_column()

    def crop(self):
        self.__remove_last_column()
        self.__remove_first_column()
        self.__remove_first_line()
        self.__remove_last_line()

    def __add_black_edges(self):
        im = np.zeros((self.height()+2, self.width()+2), dtype=np.int)
        im[1:self.height()+1, 1:self.width()+1] = self.image
        self.image = im
