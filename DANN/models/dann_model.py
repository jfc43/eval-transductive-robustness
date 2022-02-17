import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds


class DANNModel(nn.Module):
    def __init__(self, mean, std, g_num=2, f_num=3):
        super(DANNModel, self).__init__()
        self.h_dim_1 = 64
        self.h_dim = 128
        self.output_dim = 256

        # Encoder
        self.feature = nn.Sequential()
        self.feature.add_module("normalize", NormalizeLayer(mean, std))
        self.feature.add_module('g_conv1', nn.Conv2d(1, self.h_dim_1, kernel_size=5))
        self.feature.add_module('g_bn1', nn.BatchNorm2d(self.h_dim_1))
        self.feature.add_module('g_pool1', nn.MaxPool2d(2))
        self.feature.add_module('g_relu1', nn.ReLU(True))
        self.feature.add_module('g_conv2', nn.Conv2d(self.h_dim_1, self.h_dim, kernel_size=5))
        self.feature.add_module('g_bn2', nn.BatchNorm2d(self.h_dim))
        self.feature.add_module('g_drop1', nn.Dropout2d())
        self.feature.add_module('g_pool2', nn.MaxPool2d(2))
        self.feature.add_module('g_relu2', nn.ReLU(True))
        for i in range(g_num):
            self.feature.add_module('g_conv'+str(i+3), nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, padding=1))
            self.feature.add_module('g_bn'+str(i+3), nn.BatchNorm2d(self.h_dim))
            self.feature.add_module('g_relu'+str(i+3), nn.ReLU(True))

        # Discriminator
        self.feature_d = nn.Sequential()
        for i in range(5):
            self.feature_d.add_module('df_conv'+str(i), nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, padding=1))
            self.feature_d.add_module('df_relu'+str(i), nn.ReLU(True))
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.h_dim * 4 * 4, self.output_dim))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.output_dim, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # Predictor
        self.feature_f = nn.Sequential()
        for i in range(f_num):
            self.feature_f.add_module('f_conv'+str(i), nn.Conv2d(self.h_dim, self.h_dim, kernel_size=3, padding=1))
            self.feature_f.add_module('f_bn'+str(i), nn.BatchNorm2d(self.h_dim))
            self.feature_f.add_module('f_relu'+str(i), nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.h_dim * 4 * 4, self.output_dim))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(self.output_dim))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(self.output_dim, 10))

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data, alpha=0.1, return_domain_output=False):

        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        feature_d = self.feature_d(reverse_feature).view(-1, self.h_dim * 4 * 4)
        domain_output = self.domain_classifier(feature_d)

        feature = self.feature_f(feature).view(-1, self.h_dim * 4 * 4)
        class_output = self.class_classifier(feature)

        if return_domain_output:
            return class_output, domain_output
        else:
            return class_output

