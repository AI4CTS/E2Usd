import numpy
import sys
import os
sys.path.append(os.path.dirname(__file__))
import utils
import networks
import math
import torch


class BasicEncoder():
    def encode(self, X):
        pass

    def save(self, X):
        pass

    def load(self, X):
        pass


class DDEM(BasicEncoder):
    def __init__(self, win_size, batch_size, nb_steps, lr,
                 channels, depth, reduced_size, out_channels, kernel_size,
                 in_channels, cuda, gpu, M, N, win_type):
        self.target_network = self.__create_network(in_channels, channels, depth, reduced_size,
                                             out_channels, kernel_size, cuda, gpu)
        self.online_network = self.__create_network(in_channels, channels, depth, reduced_size,
                                             out_channels, kernel_size, cuda, gpu)

        self.win_type = win_type
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = networks.fncc_loss.fncc_loss(
            win_size, M, N, win_type
        )
        params_to_update = [p for p in self.online_network.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params_to_update, lr=lr)

        self.loss_list = []

        with torch.no_grad():
            for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
                target_param.data.copy_(online_param.data)

    def __create_network(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):

        network = networks.network.DDEM(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        network.double()
        if cuda:
            network.cuda(gpu)
        return network

    def fit(self, X, save_memory=False, verbose=False):
        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        train_torch_dataset = utils.Dataset(X)

        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )
        i = 0
        while i < self.nb_steps:

            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                loss = self.loss(batch, self.online_network, self.target_network, save_memory=False)
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break

        return self.online_network

    def encode(self, X, batch_size=500):
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.online_network = self.online_network.eval()

        count = 0
        with torch.no_grad():
            for batch in test_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                features[
                count * batch_size: (count + 1) * batch_size
                ] = self.online_network(batch)[0].cpu()
                count += 1

        return features

    def encode_window(self, X, win_size=128, batch_size=500, window_batch_size=10000, step=10):
        num_batch, num_channel, length = numpy.shape(X)
        num_window = int((length - win_size) / step) + 1
        embeddings = numpy.empty((num_batch, self.out_channels, num_window))

        for b in range(num_batch):
            for i in range(math.ceil(num_window / window_batch_size)):
                masking = numpy.array([X[b, :, j:j + win_size] for j in range(step * i * window_batch_size,
                                                                              step * min((i + 1) * window_batch_size,
                                                                                         num_window), step)])
                embeddings[b, :, i * window_batch_size: (i + 1) * window_batch_size] = numpy.swapaxes(
                    self.encode(masking[:], batch_size=batch_size), 0, 1)
        return embeddings[0].T

    def set_params(self, compared_length, batch_size, nb_steps, lr,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, batch_size,
            nb_steps, lr, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self

