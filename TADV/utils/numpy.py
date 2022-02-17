import numpy
import scipy.stats
import math


def one_hot(array, N):
    """
    Convert an array of numbers to an array of one-hot vectors.

    :param array: classes to convert
    :type array: numpy.ndarray
    :param N: number of classes
    :type N: int
    :return: one-hot vectors
    :rtype: numpy.ndarray
    """

    array = array.astype(int)
    assert numpy.max(array) < N
    assert numpy.min(array) >= 0

    one_hot = numpy.zeros((array.shape[0], N))
    one_hot[numpy.arange(array.shape[0]), array] = 1
    return one_hot


def expand_as(array, array_as):
    """
    Expands the tensor using view to allow broadcasting.

    :param array: input tensor
    :type array: numpy.ndarray
    :param array_as: reference tensor
    :type array_as: torch.Tensor or torch.autograd.Variable
    :return: tensor expanded with singelton dimensions as tensor_as
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    shape = list(array.shape)
    for i in range(len(array.shape), len(array_as.shape)):
        shape.append(1)

    return array.reshape(shape)


def concatenate(array1, array2, axis=0):
    """
    Basically a wrapper for numpy.concatenate, with the exception
    that the array itself is returned if its None or evaluates to False.

    :param array1: input array or None
    :type array1: mixed
    :param array2: input array
    :type array2: numpy.ndarray
    :param axis: axis to concatenate
    :type axis: int
    :return: concatenated array
    :rtype: numpy.ndarray
    """

    assert isinstance(array2, numpy.ndarray)
    if array1 is not None:
        assert isinstance(array1, numpy.ndarray)
        return numpy.concatenate((array1, array2), axis=axis)
    else:
        return array2
