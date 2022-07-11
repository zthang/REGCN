import random

from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os
import numpy as np
path_dir = os.getcwd()

class ConvTransR(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):
        super(ConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_relations*2)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, train_sample_num=None, faiss_list=None, index=None, epoch=None, nodes_id=None, mode="train", negative_rate=0):

        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        # if mode=="train":
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        # else:
        #     e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        #     e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
        if epoch !=None and epoch >= 19 and mode == "train":
            stacked_inputs_faiss = stacked_inputs.view(-1, 400).detach().cpu()
            stacked_inputs_faiss = torch.cat((torch.ones((stacked_inputs_faiss.size(0), 1))*train_sample_num, stacked_inputs_faiss, triplets[:, 1].view(-1, 1).detach().cpu()), dim=1)
            faiss_list.append(stacked_inputs_faiss.numpy())
            print(f"epoch:{epoch}, rel_list insert_num:{stacked_inputs_faiss.size(0)}")
        if mode == "test" and index != None:
            k = 100
            stacked_inputs_faiss = stacked_inputs.view(-1, 400).detach().cpu()
            D, I = index.search(stacked_inputs_faiss.numpy(), k)
            nearest_neighbors = faiss_list[I]
            retrieval_label = nearest_neighbors[:, :, -1].astype(int)
            label_distribution = np.apply_along_axis(lambda x: np.bincount(x, minlength=emb_rel.size(0)), axis=1, arr=retrieval_label)
            label_distribution = label_distribution/k
            label_distribution = torch.tensor(label_distribution)
            label_distribution = label_distribution.to(stacked_inputs.device)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_rel.transpose(1, 0))
        if mode == "test" and index != None:
            x = (x + label_distribution)/2
        return x


class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()
        # 初始化relation embeddings
        # self.emb_rel = torch.nn.Embedding(num_relations, embedding_dim, padding_idx=0)

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, train_sample_num=None, faiss_list=None, index=None, epoch=None, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        if epoch != None and epoch >= 19 and mode == "train":
            stacked_inputs_faiss = stacked_inputs.view(-1, 400).detach().cpu()
            stacked_inputs_faiss = torch.cat((torch.ones((stacked_inputs_faiss.size(0), 1))*train_sample_num, stacked_inputs_faiss, triplets[:, 2].view(-1, 1).detach().cpu()), dim=1)
            faiss_list.append(stacked_inputs_faiss.numpy())
            print(f"epoch:{epoch}, entity_list insert_num:{stacked_inputs_faiss.size(0)}")
        if mode == "test" and index != None:
            k = 100
            stacked_inputs_faiss = stacked_inputs.view(-1, 400).detach().cpu()
            D, I = index.search(stacked_inputs_faiss.numpy(), k)
            nearest_neighbors = faiss_list[I]
            retrieval_label = nearest_neighbors[:, :, -1].astype(int)
            label_distribution = np.apply_along_axis(lambda x: np.bincount(x, minlength=e1_embedded_all.size(0)), axis=1, arr=retrieval_label)
            label_distribution = label_distribution/k
            label_distribution = torch.tensor(label_distribution)
            label_distribution = label_distribution.to(stacked_inputs.device)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
            if mode == "test" and index != None:
                x = (x + label_distribution)/2
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        return x

    def forward_slow(self, embedding, emb_rel, triplets):

        e1_embedded_all = F.tanh(embedding)
        # e1_embedded_all = embedding
        batch_size = len(triplets)
        e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
        # translate to sub space
        # e1_embedded = torch.matmul(e1_embedded, sub_trans)
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        e2_embedded = e1_embedded_all[triplets[:, 2]]
        score = torch.sum(torch.mul(x, e2_embedded), dim=1)
        pred = score
        return pred