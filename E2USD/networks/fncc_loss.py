import numpy as np
import torch



class fncc_loss(torch.nn.modules.loss._Loss):

    def __init__(self, win_size, M, N, win_type):
        super(fncc_loss, self).__init__()
        self.win_size = win_size
        self.win_type = win_type
        self.M = M
        self.N = N
        self.total = 0
        self.total_condition = 0

    def forward(self, batch, encoder, save_memory=False):
        M = self.M
        N = self.N
        length_pos_neg = self.win_size
        total_length = batch.size(2)
        center_list = []
        center_trend_list = []
        center_seasonal_list = []
        loss1 = 0

        total_embeddings=[]
        total_trend_embeddings=[]
        total_seasonal_embeddings=[]

        for i in range(M):
            random_pos = np.random.randint(0, high=total_length - length_pos_neg * 2 + 1, size=1)
            rand_samples = [batch[0, :, i: i + length_pos_neg] for i in range(random_pos[0], random_pos[0] + N)]

            intra_sample =torch.stack(rand_samples)

            embeddings, trend_x_embedding, seasonal_x_embedding = encoder(intra_sample) #([4, 4]) N / embedding_channel
            total_embeddings.append(embeddings)
            total_trend_embeddings.append(trend_x_embedding)
            total_seasonal_embeddings.append(seasonal_x_embedding)

            size_representation = embeddings.size(1)

            for i in range(N):
                for j in range(N):
                    if j <= i:
                        continue
                    else:

                        similarity_embedding = torch.bmm(
                            embeddings[i].view(1, 1, size_representation),
                            embeddings[j].view(1, size_representation, 1))
                        loss1_term =-torch.mean(torch.nn.functional.logsigmoid(
                            similarity_embedding))
                        loss1 += loss1_term


            center = torch.mean(embeddings, dim=0)
            center_trend = torch.mean(trend_x_embedding, dim=0)
            center_seasonal = torch.mean(seasonal_x_embedding, dim=0)
            center_list.append(center)
            center_seasonal_list.append(center_seasonal)
            center_trend_list.append(center_trend)

        loss2 = 0
        smi = []
        loss2_item = []
        totalnumber=0
        for i in range(M):
            for ii in range(N):
                for j in range(M):
                    if j <= i:
                        continue
                    for jj in range(N):
                        totalnumber+=1
                        similarity_trend = torch.bmm(
                            total_trend_embeddings[i][ii].view(1, 1, size_representation),
                            total_trend_embeddings[j][jj].view(1, size_representation, 1))
                        similarity_seasonal = torch.bmm(
                            total_seasonal_embeddings[i][ii].view(1, 1, size_representation),
                            total_seasonal_embeddings[j][jj].view(1, size_representation, 1))

                        loss2_term = torch.bmm(
                            total_embeddings[i][ii].view(1, 1, size_representation),
                            total_embeddings[j][jj].view(1, size_representation, 1))

                        smi_value = similarity_trend * similarity_seasonal
                        smi.append(smi_value.item())
                        loss2_item.append(loss2_term)

        sorted_indices = sorted(range(len(smi)), key=lambda k: smi[k])
        half_index = len(sorted_indices) // 2
        indices_of_smallest_half = sorted_indices[:half_index]

        for idx in indices_of_smallest_half:
            loss2 += loss2_item[idx]

        loss1 = (loss1)/ (M * N * (N - 1) / 2)
        loss2 = (loss2)/ (totalnumber / 2)
        loss = loss1 + loss2
        return loss

