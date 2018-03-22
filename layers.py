from torch import nn, torch
from torch.autograd import Variable
from torch.nn import Parameter, init


class GaussianNoise(nn.Module):

    def __init__(self, stddev, mean=.0):
        """
        Additive Gaussian Noise layer
        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean

    def forward(self, x):
        if self.training:
            noise = Variable(x.data.new(x.size()).normal_(self.mean,
                                                          self.stddev))
            return x + noise
        return x


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m


class SelfAttention(nn.Module):

    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax()

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        init.uniform(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores
