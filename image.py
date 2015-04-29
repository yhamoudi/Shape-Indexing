import sys
import re # regular expressions
import numpy as np
import scipy.misc
from matplotlib import pyplot

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
        # boolean matrix : False if black, True if white
        # (the picture itself is white) :
        self.image = np.vectorize(lambda x: x != 0)(image)

    def height(self):
        return self.image.shape[0]

    def width(self):
        return self.image.shape[1]

    def noise(self):
      self.image = self.image + 0.2 * np.random.randn(*self.image.shape)**2
      self.image = np.vectorize(lambda x: x > 0.5)(self.image)

    def normalize(self):
        self.__reverse_colors()  # reverse colors if the picture itself was white
        self.__crop()            # crop the image until all borders contain a white pixel
        self.__resize()          # resize the image
        self.__add_black_edges() # add a black border all around the image

    def __reverse_colors(self):
        if self.image[0,:].sum() + self.image[self.height()-1,:].sum() + self.image[:,0].sum() + self.image[:, self.width()-1].sum() > self.height() + self.width():
          self.image = np.vectorize(lambda x: x == 0)(self.image)

    def __resize(self, max_edge_size=50):
        alpha = float(max_edge_size) / float(max(self.height(), self.width()))
        self.image = scipy.misc.imresize(self.image, alpha)
        self.image = np.vectorize(lambda x: x > 0.5)(self.image)

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

    def __crop(self):
        self.__remove_last_column()
        self.__remove_first_column()
        self.__remove_first_line()
        self.__remove_last_line()

    def __add_black_edges(self):
        im = np.zeros((self.height()+2, self.width()+2), dtype=np.int)
        im[1:self.height()+1, 1:self.width()+1] = self.image
        self.image = im

    def print(self):
        pyplot.imshow(self.image, pyplot.cm.gray)
        pyplot.show()
