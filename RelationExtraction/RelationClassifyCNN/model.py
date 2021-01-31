import torch
import torch.nn as nn
from torch.nn import init


class CNN(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis

        self.dropout_value = config.dropout
        self.filter_num = config.filter_num
        self.window = config.window
        self.hidden_size = config.hidden_size

        self.dim = self.word_dim + 2 * self.pos_dim

        # load the pre-train model for word features
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        # embedding for features for position distance relative to entity 1
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        # embedding for features for position distance  relative to entity 2
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=False,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )

        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(self.dropout_value)

        self.linear = nn.Linear(
            in_features=self.filter_num,
            out_features=self.hidden_size,
            bias=False
        )
        self.dense = nn.Linear(
            in_features=self.hidden_size + 6 * self.word_dim,
            out_features=self.class_num,
            bias=False
        )

        # initialize weight
        init.xavier_normal_(self.pos1_embedding.weight)
        init.xavier_normal_(self.pos2_embedding.weight)
        init.xavier_normal_(self.conv.weight)
        # init.constant_(self.conv.bias, 0.)
        init.xavier_normal_(self.linear.weight)
        # init.constant_(self.linear.bias, 0.)
        init.xavier_normal_(self.dense.weight)
        # init.constant_(self.dense.bias, 0.)

    def encoder_layer(self, token, pos1, pos2):
        word_emb = self.word_embedding(token)  # B*L*word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb], dim=-1)
        return emb  # B*L*D, D=word_dim+2*pos_dim

    def conv_layer(self, emb, mask):
        emb = emb.unsqueeze(dim=1)  # B*1*L*D
        conv = self.conv(emb)  # B*C*L*1

        # mask, remove the effect of 'PAD'
        conv = conv.view(-1, self.filter_num, self.max_len)  # B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L
        mask = mask.expand(-1, self.filter_num, -1)  # B*C*L
        conv = conv.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        conv = conv.unsqueeze(dim=-1)  # B*C*L*1
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1
        pool = pool.view(-1, self.filter_num)  # B*C
        return pool

    def forward(self, data):
        token = data[0][:, 0, :].view(-1, self.max_len)
        pos1 = data[0][:, 1, :].view(-1, self.max_len)
        pos2 = data[0][:, 2, :].view(-1, self.max_len)
        mask = data[0][:, 3, :].view(-1, self.max_len)
        lexical = data[1].view(-1, 6)

        lexical_emb = self.word_embedding(lexical)
        lexical_emb = lexical_emb.view(-1, self.word_dim * 6)

        emb = self.encoder_layer(token, pos1, pos2)
        emb = self.dropout(emb)
        conv = self.conv_layer(emb, mask)
        pool = self.single_maxpool_layer(conv)

        sentence_feature = self.linear(pool)
        sentence_feature = self.tanh(sentence_feature)
        sentence_feature = self.dropout(sentence_feature)

        features = torch.cat((lexical_emb, sentence_feature), 1)
        logits = self.dense(features)
        return logits