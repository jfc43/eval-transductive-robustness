import torch
import numpy
import scipy.ndimage
import math
import random


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def is_cuda(mixed):
    """
    Check if model/tensor is on CUDA.

    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.autograd.Variable or torch.Tensor
    :return: on cuda
    :rtype: bool
    """

    assert isinstance(mixed, torch.nn.Module) or isinstance(mixed, torch.autograd.Variable) \
        or isinstance(mixed, torch.Tensor), 'mixed has to be torch.nn.Module, torch.autograd.Variable or torch.Tensor'

    is_cuda = False
    if isinstance(mixed, torch.nn.Module):
        is_cuda = True
        for parameters in list(mixed.parameters()):
            is_cuda = is_cuda and parameters.is_cuda
    if isinstance(mixed, torch.autograd.Variable):
        is_cuda = mixed.is_cuda
    if isinstance(mixed, torch.Tensor):
        is_cuda = mixed.is_cuda

    return is_cuda

def binary_labels(classes):
    """
    Convert 0,1 labels to -1,1 labels.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    """

    classes[classes == 0] = -1
    return classes


def one_hot(classes, C):
    """
    Convert class labels to one-hot vectors.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    :param C: number of classes
    :type C: int
    :return: one hot vector as B x C
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(classes, torch.autograd.Variable) or isinstance(classes, torch.Tensor), 'classes needs to be torch.autograd.Variable or torch.Tensor'
    assert len(classes.size()) == 2 or len(classes.size()) == 1, 'classes needs to have rank 2 or 1'
    assert C > 0

    if len(classes.size()) < 2:
        classes = classes.view(-1, 1)

    one_hot = torch.Tensor(classes.size(0), C)
    if is_cuda(classes):
         one_hot = one_hot.cuda()

    if isinstance(classes, torch.autograd.Variable):
        one_hot = torch.autograd.Variable(one_hot)

    one_hot.zero_()
    one_hot.scatter_(1, classes, 1)

    return one_hot


def tensor_or_value(mixed):
    """
    Get tensor or single value.

    :param mixed: variable, tensor or value
    :type mixed: mixed
    :return: tensor or value
    :rtype: torch.Tensor or value
    """

    if isinstance(mixed, torch.Tensor):
        if mixed.numel() > 1:
            return mixed
        else:
            return mixed.item()
    elif isinstance(mixed, torch.autograd.Variable):
        return tensor_or_value(mixed.cpu().data)
    else:
        return mixed


def as_variable(mixed, cuda=False, requires_grad=False):
    """
    Get a tensor or numpy array as variable.

    :param mixed: input tensor
    :type mixed: torch.Tensor or numpy.ndarray
    :param device: gpu or not
    :type device: bool
    :param requires_grad: gradients
    :type requires_grad: bool
    :return: variable
    :rtype: torch.autograd.Variable
    """

    assert isinstance(mixed, numpy.ndarray) or isinstance(mixed, torch.Tensor), 'input needs to be numpy.ndarray or torch.Tensor'

    if isinstance(mixed, numpy.ndarray):
        mixed = torch.from_numpy(mixed)

    if cuda:
        mixed = mixed.cuda()
    return torch.autograd.Variable(mixed, requires_grad)


def tile(a, dim, n_tile):
    """
    Numpy-like tiling in torch.

    :param a: tensor
    :type a: torch.Tensor or torch.autograd.Variable
    :param dim: dimension to tile
    :type dim: int
    :param n_tile: number of tiles
    :type n_tile: int
    :return: tiled tensor
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(numpy.concatenate([init_dim * numpy.arange(n_tile) + i for i in range(init_dim)]))
    if is_cuda(a):
        order_index = order_index.cuda()
    return torch.index_select(a, dim, order_index)


def expand_as(tensor, tensor_as):
    """
    Expands the tensor using view to allow broadcasting.

    :param tensor: input tensor
    :type tensor: torch.Tensor or torch.autograd.Variable
    :param tensor_as: reference tensor
    :type tensor_as: torch.Tensor or torch.autograd.Variable
    :return: tensor expanded with singelton dimensions as tensor_as
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    view = list(tensor.size())
    for i in range(len(tensor.size()), len(tensor_as.size())):
        view.append(1)

    return tensor.view(view)


def get_exponential_scheduler(optimizer, batches_per_epoch, gamma=0.97):
    """
    Get exponential scheduler.

    Note that the resulting optimizer's step function is called after each batch!

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param batches_per_epoch: number of batches per epoch
    :type batches_per_epoch: int
    :param gamma: gamma
    :type gamma: float
    :return: scheduler
    :rtype: torch.optim.lr_scheduler.LRScheduler
    """

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: gamma ** math.floor(epoch/batches_per_epoch)])


def classification_error(logits, targets, reduction='mean'):
    """
    Accuracy.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduce: reduce to number or keep per element
    :type reduce: bool
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        values, indices = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)
    else:
        indices = torch.round(torch.sigmoid(logits)).view(-1)

    errors = torch.clamp(torch.abs(indices.long() - targets.long()), max=1)
    if reduction == 'mean':
        return torch.mean(errors.float())
    elif reduction == 'sum':
        return torch.sum(errors.float())
    else:
        return errors


def f7p_loss(logits, true_classes, reduction='mean'):
    if logits.size(1) > 1:
        current_probabilities = torch.nn.functional.softmax(logits, dim=1)
        current_probabilities = current_probabilities * (1 - one_hot(true_classes, current_probabilities.size(1)))
        loss = torch.max(current_probabilities, dim=1)[0]
    else:
        loss = true_classes.float()*(1 - torch.nn.functional.sigmoid(logits.view(-1))) + (1 - true_classes.float())*(torch.nn.functional.sigmoid(logits.view(-1)))

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    else:
        raise ValueError


def classification_loss(logits, targets, reduction='mean'):
    """
    Calculates either the multi-class or binary cross-entropy loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    if logits.size()[1] > 1:
        return torch.nn.functional.cross_entropy(logits, targets, reduction=reduction)
    else:
        # probability 1 is class 1
        # probability 0 is class 0
        return torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits).view(-1), targets.float(), reduction=reduction)


def step_function_perturbation(perturb, epsilon_0, alpha=1e-4, norm_type='inf', smooth_approx=False, temperature=0.01):
    """
    Step function applied to the perturbation norm. By default, it computes the exact step function which is not
    differentiable. If a smooth approximation based on the sigmoid function is needed, set `smooth_approx=True` and set
    the `temperature` to a suitably small value.

    :param perturb: Torch Tensor with the perturbation. Can be a tensor of shape `(b, d1, ...)`, where `b` is the batch
                    size and the rest are dimensions. Can also be a single vector of shape `[d]`.
    :param epsilon_0: Radius of the smaller perturbation ball - a small positive value.
    :param alpha: Small negative offset for the step function. The step function's value is `-alpha` when the
                  perturbation norm is less than `epsilon_0`.
    :param norm_type: Type of norm: 'inf' for infinity norm, '1', '2' etc for the other types of norm.
    :param smooth_approx: Set to True to get a sigmoid-based approximation of the step function.
    :param temperature: small non-negative value that controls  the steepness of the sigmoid approximation.

    :returns: tensor of function values computed for each element in the batch. Has shape `[b]`.
    """
    assert isinstance(perturb, (torch.Tensor, torch.autograd.Variable)), ("Input 'perturb' should be of type "
                                                                          "torch.Tensor or torch.autograd.Variable")
    s = perturb.shape
    dim = 1
    if len(s) > 2:
        perturb = perturb.view(s[0], -1)    # flatten into a vector
    elif len(s) == 1:
        # single vector
        dim = None

    if norm_type == 'inf':
        norm_type = float('inf')
    elif not isinstance(norm_type, (int, float)):
        # example: norm_type = '2'
        norm_type = int(norm_type)

    norm_val = torch.linalg.vector_norm(perturb, ord=norm_type, dim=dim)
    if not smooth_approx:
        return torch.where(norm_val <= epsilon_0, -1. * alpha, 1.)
    else:
        return torch.sigmoid((1. / temperature) * (norm_val - epsilon_0)) - alpha


def ramp_function_perturbation(perturb, epsilon_0, epsilon, alpha=1e-4, norm_type='inf'):
    """
    Ramp function applied to the perturbation norm as defined in the paper.

    :param perturb: Torch Tensor with the perturbation. Can be a tensor of shape `(b, d1, ...)`, where `b` is the batch
                    size and the rest are dimensions. Can also be a single vector of shape `(d)`.
    :param epsilon_0: Radius of the smaller perturbation ball - a small positive value.
    :param epsilon: Radius of the larger perturbation ball. Should be >= `epsilon_0`.
    :param alpha: Small negative offset for the step function. The step function's value is `-alpha` when the
                  perturbation norm is less than `epsilon_0`.
    :param norm_type: Type of norm: 'inf' for infinity norm, '1', '2' etc for the other types of norm.

    :returns: tensor of function values computed for each element in the batch. Has shape `[b]`.
    """
    assert isinstance(perturb, (torch.Tensor, torch.autograd.Variable)), ("Input 'perturb' should be of type "
                                                                          "torch.Tensor or torch.autograd.Variable")
    assert epsilon >= epsilon_0, "Value of 'epsilon' cannot be smaller than 'epsilon_0'"
    s = perturb.shape
    dim = 1
    if len(s) > 2:
        perturb = perturb.view(s[0], -1)    # flatten into a vector
    elif len(s) == 1:
        # single vector
        dim = None

    if norm_type == 'inf':
        norm_type = float('inf')
    elif not isinstance(norm_type, (int, float)):
        # example: norm_type = '2'
        norm_type = int(norm_type)

    norm_val = torch.linalg.vector_norm(perturb, ord=norm_type, dim=dim)
    temp = torch.maximum(norm_val - epsilon_0, torch.zeros_like(norm_val))

    return ((1. + alpha) / (epsilon - epsilon_0)) * temp - alpha


def max_p_loss(logits, targets=None, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    max_log = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)[0]
    if reduction == 'mean':
        return torch.mean(max_log)
    elif reduction == 'sum':
        return torch.sum(max_log)
    else:
        return max_log


def max_log_loss(logits, targets=None, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    max_log = torch.max(torch.nn.functional.log_softmax(logits, dim=1), dim=1)[0]
    if reduction == 'mean':
        return torch.mean(max_log)
    elif reduction == 'sum':
        return torch.sum(max_log)
    else:
        return max_log


def cross_entropy_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = torch.sum(- targets * torch.nn.functional.log_softmax(logits, dim=1), dim=1)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


class View(torch.nn.Module):
    """
    Simple view layer.
    """

    def __init__(self, *args):
        """
        Constructor.

        :param args: shape
        :type args: [int]
        """

        super(View, self).__init__()

        self.shape = args

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(self.shape)


class Flatten(torch.nn.Module):
    """
    Flatten module.
    """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(input.shape[0], -1)


class Clamp(torch.nn.Module):
    """
    Wrapper for clamp.
    """

    def __init__(self, min=0, max=1):
        """
        Constructor.
        """

        super(Clamp, self).__init__()

        self.min = min
        """ (float) Min value. """

        self.max = max
        """ (float) Max value. """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.clamp(torch.clamp(input, min=self.min), max=self.max)


class Scale(torch.nn.Module):
    """
    Simply scaling layer, mainly to allow simple saving and loading.
    """

    def __init__(self, shape):
        """
        Constructor.

        :param shape: shape
        :type shape: [int]
        """

        super(Scale, self).__init__()

        self.weight = torch.nn.Parameter(torch.zeros(shape)) # min
        self.bias = torch.nn.Parameter(torch.ones(shape)) # max

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return expand_as(self.weight, input) + torch.mul(expand_as(self.bias, input) - expand_as(self.weight, input), input)


class Entropy(torch.nn.Module):
    """
    Entropy computation based on logits.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(Entropy, self).__init__()

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return -1.*torch.sum(torch.nn.functional.softmax(input, dim=1) * torch.nn.functional.log_softmax(input, dim=1))


class Normalize(torch.nn.Module):
    """
    Normalization layer to be learned.
    """

    def __init__(self, n_channels):
        """
        Constructor.

        :param n_channels: number of channels
        :type n_channels: int
        """

        super(Normalize, self).__init__()

        self.weight = torch.nn.Parameter(torch.ones(n_channels))
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return (input - self.bias.view(1, -1, 1, 1))/self.weight.view(1, -1, 1, 1)


class GaussianLayer(torch.nn.Module):
    """
    Gaussian convolution.

    """

    def __init__(self, sigma=3, channels=3):
        """

        """
        super(GaussianLayer, self).__init__()

        self.sigma = sigma
        """ (float) Sigma. """

        padding = math.ceil(self.sigma)
        kernel = 2*padding + 1

        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((padding, padding, padding, padding)),
            torch.nn.Conv2d(channels, channels, kernel, stride=1, padding=0, bias=None, groups=channels)
        )

        n = numpy.zeros((kernel, kernel))
        n[padding, padding] = 1

        k = scipy.ndimage.gaussian_filter(n, sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return self.seq(input)
