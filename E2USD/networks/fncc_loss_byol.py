import numpy as np
import torch
import random

def add_noise(data, mean=0, std=0.05):
    noise = torch.randn(data.size()) * std + mean
    return data + noise

def apply_scaling(data, mean=1.0, std=0.1):
    scaling_factor = torch.normal(mean=mean, std=std, size=data.size())
    return data * scaling_factor

# 随机选择数据增强方法




class fncc_loss(torch.nn.modules.loss._Loss):

    def __init__(self, win_size, M, N, win_type):
        super(fncc_loss, self).__init__()
        self.win_size = win_size
        self.win_type = win_type
        self.M = M
        self.N = N
        self.total = 0
        self.total_condition = 0

    def forward(self, batch, encoder, encoder2, save_memory=False):
        M = self.M
        N = self.N
        length_pos_neg = self.win_size
        total_length = batch.size(2)

        loss1 = 0


        for i in range(M):
            random_pos = int(np.random.randint(0, high=total_length - length_pos_neg * 2 + 1, size=1))
            rand_samples = [batch[0, :, random_pos: random_pos + length_pos_neg],batch[0, :, random_pos: random_pos + length_pos_neg],batch[0, :, random_pos: random_pos + length_pos_neg],batch[0, :, random_pos: random_pos + length_pos_neg]]

            intra_sample =torch.stack(rand_samples)
            embeddings, trend_x_embedding, seasonal_x_embedding = encoder(intra_sample) #([4, 4]) N / embedding_channel
            if random.choice([True, False]):
                # 应用噪声增强
                intra_sample_augmented = add_noise(intra_sample)
            else:
                # 应用缩放增强
                intra_sample_augmented = apply_scaling(intra_sample)
            embeddings_augmented, trend_x_embedding_augmented, seasonal_x_embedding_augmented = encoder(intra_sample_augmented) #([4, 4]) N / embedding_channel

            size_representation = embeddings.size(1)

            for i in range(N):
                for j in range(N):
                    if j <= i:
                        continue
                    else:

                        similarity_embedding = torch.bmm(
                            embeddings[i].view(1, 1, size_representation),
                            embeddings_augmented[j].view(1, size_representation, 1))
                        loss1_term =-torch.mean(torch.nn.functional.logsigmoid(
                            similarity_embedding))
                        loss1 += loss1_term


        loss1 = (loss1)/ (M * N * (N - 1) / 2)
        loss = loss1
        return loss

