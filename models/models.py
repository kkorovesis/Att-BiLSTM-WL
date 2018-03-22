from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers import GaussianNoise, SelfAttention
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, embeddings, num_classes, **kwargs):

        super(RNN, self).__init__()

        rnn_hidden_size = kwargs.get("rnn_size", 150)
        num_rnn_layers = kwargs.get("num_rnn_layers", 2)
        bidirectional = kwargs.get("bidirectional", True)
        noise = kwargs.get("noise", 0.5)
        dropout_embeds = kwargs.get("dropout_embeds", 0.5)
        dropout_rnn = kwargs.get("dropout_rnn", 0.5)
        trainable_emb = kwargs.get("trainable_emb", False)

        self.embedding = nn.Embedding(num_embeddings=embeddings.shape[0], embedding_dim=embeddings.shape[1])
        self.noise_emb = GaussianNoise(noise)
        self.init_embeddings(embeddings, trainable_emb)
        self.dropout_embeds = nn.Dropout(dropout_embeds)
        self.dropout_rnn = nn.Dropout(dropout_rnn)
        self.batch_size = 128
        self.seed = 1111

        self.shared_lstm = nn.LSTM(input_size=embeddings.shape[1],
                                  hidden_size=rnn_hidden_size,
                                  num_layers=num_rnn_layers,
                                  bidirectional=bidirectional,
                                  dropout=dropout_rnn,
                                  batch_first=True)

        if bidirectional:
            rnn_hidden_size *= 4
        else:
            rnn_hidden_size *= 2

        self.attention = SelfAttention(attention_size=rnn_hidden_size, batch_first=True)

        self.linear = nn.Linear(rnn_hidden_size, num_classes)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def last_timestep(self, rnn, h):
        if rnn.bidirectional:
            return torch.cat((h[-2], h[-1]), 1)
        else:
            return h[-1]

    def forward(self, message, topic, lengths, topic_lengths):

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        ###MESSAGE MODEL###

        embeds = self.embedding(message)
        embeds = self.noise_emb(embeds)
        embeds = self.dropout_embeds(embeds)

        # pack the batch
        embeds_pckd = pack_padded_sequence(embeds, list(lengths.data), batch_first=True)

        mout_pckd, (hx1, cx1) = self.shared_lstm(embeds_pckd)

        # unpack output - no need if we are going to use only the last outputs
        mout_unpckd, _ = pad_packed_sequence(mout_pckd, batch_first=True)  # [batch_size,seq_length,300]

        # Last timestep output is not used
        # message_output = self.last_timestep(self.shared_lstm, hx1)
        # message_output = self.dropout_rnn(message_output)

        ###TOPIC MODEL###

        topic_embeds = self.embedding(topic)
        topic_embeds = self.dropout_embeds(topic_embeds)

        tout, (hx2, cx2) = self.shared_lstm(topic_embeds)
        tout = self.dropout_rnn(tout)

        mask = (topic > 0).float().unsqueeze(-1)
        tout = torch.sum(tout * mask, dim=1)
        tout = tout / topic_lengths.unsqueeze(-1).float()
        tout = torch.unsqueeze(tout, 1)
        tout = tout.expand(mout_unpckd.size(0), mout_unpckd.size(1), mout_unpckd.size(2))

        out = torch.cat((mout_unpckd, tout), 2)
        representations, attentions = self.attention(out, lengths)

        return self.linear(representations)


class CNNClassifier(nn.Module):

    def __init__(self, embeddings, num_classes, **kwargs):
        super(CNNClassifier, self).__init__()

        kernel_dim = kwargs.get("kernel_dim", 100)
        kernel_sizes = kwargs.get("kernel_sizes", (3, 4, 5))
        dropout = kwargs.get("dropout", 0.5)
        trainable_emb = kwargs.get("trainable_emb", False)
        noise_emb = kwargs.get("noise", 0.5)

        self.drop_emb = nn.Dropout(dropout)
        self.noise_emb = GaussianNoise(noise_emb)
        self.embedding = nn.Embedding(num_embeddings=embeddings.shape[0], embedding_dim=embeddings.shape[1])
        self.init_embeddings(embeddings, trainable_emb)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, 400)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, num_classes)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def forward(self, inputs, aspects, lengths, aspects_lengths):

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        inputs = self.embedding(inputs)
        inputs = self.noise_emb(inputs)
        inputs = self.drop_emb(inputs)

        aspects = self.embedding(aspects)
        aspects = self.drop_emb(aspects)

        mask = (aspects > 0).float()
        aspects = torch.sum(aspects * mask, dim=1)
        new_asp = aspects / aspects_lengths.unsqueeze(-1).float()
        new_asp = torch.unsqueeze(new_asp, 1)
        new_asp = new_asp.expand(inputs.size(0), inputs.size(1), inputs.size(2))

        concat = torch.cat((inputs, new_asp), 2)
        inputs = concat.unsqueeze(1)

        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]
        concatenated = torch.cat(inputs, 1)

        concatenated = self.dropout(concatenated)

        return self.fc(concatenated)

