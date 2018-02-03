#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import itertools


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # TODO: Your ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to
        # False.
        self.x = x
        self.y = y
        self.num_of_samples = x.shape[0]
        self.height = x.shape[1]
        self.width = x.shape[2]
        self.is_horizontal_flip = False
        self.is_vertical_flip = False
        self.is_add_noise = False
        self.shift_height = 0
        self.shift_width = 0
        self.degree_rotation = 0
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data indefinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        # TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.

        # One possible pseudo code for your reference:
        #######################################################################
        #   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        #   while True:
        #       if (batch_count < total number of batches possible):
        #           batch_count = batch_count + 1
        #           yield(next batch of x and y indicated by batch_count)
        #       else:
        #           shuffle(x)
        #           reset batch_count
        x = self.x
        total_number_of_batches = self.num_of_samples // batch_size
        batch_count = 0
        while True:
            if batch_count < total_number_of_batches:
                batch_count += 1
                yield(x[batch_count])
            else:
                np.random.shuffle(self.x)
                batch_count = 0
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def show(self):
        """
        Plot the top 16 images (index 0~15) of self.x for visualization.
        """
        pics = self.next_batch_gen(16, shuffle=True)
        # top16 = np.asarray(list(itertools.islice(pics, 0, 1, 1)))
        top16 = np.asarray(list(itertools.islice(pics, 0, 16, 1)))
        top16 = top16.reshape(3, 32, 32, -1).transpose(3, 1, 2, 0)
        self.visualize_pics(top16)
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return:
        """

        # TODO: Implement the translate function. Remember to record the value of the number of pixels translated.
        # Note: You may wonder what values to append to the edge after the translation. Here, use rolling instead. For
        # example, if you translate 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
        # right edge of the picture.
        # Hint: Numpy.roll
        # (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
        x = self.x
        x = np.roll(x, shift_height, axis=0)
        x = np.roll(x, shift_width, axis=1)
        self.x = x
        self.shift_height += shift_height
        self.shift_width += shift_width
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.

        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """
        # TODO: Implement the rotate function. Remember to record the value of
        # rotation degree.
        x = self.x
        x = rotate(x, 30, reshape=False)
        self.degree_rotation = angle
        self.x = x
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        """
        # TODO: Implement the flip function. Remember to record the boolean values is_horizontal_flip and
        # is_vertical_flip.
        x = self.x
        if mode == 'h':
            x = np.fliplr(x)
            self.is_horizontal_flip = True
        elif mode == 'v':
            x = np.flipud(x)
            self.is_vertical_flip = True
        elif mode == 'hv':
            x = np.fliplr(x)
            x = np.flipud(x)
            self.is_horizontal_flip = True
            self.is_vertical_flip = True
        else:
            print ("Error! Try again...")
        self.x = x
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        """
        # TODO: Implement the add_noise function. Remember to record the
        # boolean value is_add_noise. You can try uniform noise or Gaussian
        # noise or others ones that you think appropriate.
        x = self.x
        n,row,col,ch = x.shape
        mu = 0
        sigma = 0.1
        gauss = np.random.normal(mu,sigma,(n,row,col,ch))
        gauss = gauss.reshape(n,row,col,ch)
        mask = np.random.choice([0, 1], size=x.shape, p=[(1 - portion), portion])
        x = x + amplitude * mask * gauss
        self.x = x
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

    def visualize_pics(self, pics):
        ## visualization
        num_pics = pics.shape[0]
        grid_size = int(np.ceil(np.sqrt(num_pics)))
        padding = 2
        figure = np.zeros(((32+2*padding)*grid_size, (32+2*padding)*grid_size, 3))
        for r in range(grid_size):
            for c in range(grid_size):
                pid = r*grid_size + c
                if pid < num_pics:
                    pic = pics[pid]
                    high, low = np.max(pic), np.min(pic)
                    pic = 255.0*(pic-low)/(high-low)
                    rid = (32+2*padding)*r
                    cid = (32+2*padding)*c
                    figure[rid+padding:rid+padding+32, cid+padding:cid+padding+32, :] = pic

        plt.imshow(figure.astype('uint8'))
        plt.gca().axis('off')
        plt.show()
