import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class Learn_to_weight(nn.Module):
    def __init__(self, emb_dim1=256, emb_dim2=512, emb_dim3=1024):
        super(Learn_to_weight, self).__init__()
        self.emb_dim1 = emb_dim1
        self.emb_dim2 = emb_dim2
        self.emb_dim3 = emb_dim3
        self.scale1 = emb_dim1 ** -0.5
        self.scale2 = emb_dim2 ** -0.5
        self.scale3 = emb_dim3 ** -0.5

        self.proj_in1 = nn.Conv2d(emb_dim1, emb_dim1, kernel_size=1, stride=1, padding=0)
        self.proj_in2 = nn.Conv2d(emb_dim2, emb_dim2, kernel_size=1, stride=1, padding=0)
        self.proj_in3 = nn.Conv2d(emb_dim3, emb_dim3, kernel_size=1, stride=1, padding=0)

    def forward(self, query, x):
        '''
        :param query: [batch_szie, c, h, w]
        :param x: [k-shot, batch_size, c, h, w]
        :return:
        '''

        k, b, c, h, w = x[0].shape

        x[0] = rearrange(x[0], 'k b c h w -> (k b) c h w')
        k1 = self.proj_in1(x[0])   # [(k b), c, h, w] = [(k b), 256, 32, 32]

        Q1 = rearrange(query[0], 'b c h w -> b 1 (c h w)')  # [b, c, h, w] = [(k b), 256, 32, 32]
        K1 = rearrange(k1, '(k b) c h w -> b k (c h w)', b=b, k=k)
        V1 = rearrange(x[0], '(k b) c h w -> b k (c h w)', b=b, k=k)   # [batch_size, h*w, c] = [3, 262144, 512]

        # [batch_size, h*w, seq_len]
        att_weights1 = torch.bmm(K1, Q1.transpose(1, 2))  # [b, k, 1]
        att_weights1 = att_weights1 * self.scale1

        att_weights1 = F.softmax(att_weights1, dim=1)
        out1 = V1 * att_weights1  # [b, k, (c h w)]
        out1 = rearrange(out1, 'b k (c h w) -> k b c h w', c=c, h=h, w=w)
        out1 = out1.sum(dim=0)

        k, b, c, h, w = x[1].shape

        x[1] = rearrange(x[1], 'k b c h w -> (k b) c h w')
        k2 = self.proj_in2(x[1])   # [(k b), c, h, w] = [(k b), 256, 32, 32]

        Q2 = rearrange(query[1], 'b c h w -> b 1 (c h w)')  # [b, c, h, w] = [(k b), 256, 32, 32]
        K2 = rearrange(k2, '(k b) c h w -> b k (c h w)', b=b, k=k)
        V2 = rearrange(x[1], '(k b) c h w -> b k (c h w)', b=b, k=k)   # [batch_size, h*w, c] = [3, 262144, 512]

        # [batch_size, h*w, seq_len]
        att_weights2 = torch.bmm(K2, Q2.transpose(1, 2))  # [b, k, 1]
        att_weights2 = att_weights2 * self.scale2

        att_weights2 = F.softmax(att_weights2, dim=1)
        out2 = V2 * att_weights2  # [b, k, (c h w)]
        out2 = rearrange(out2, 'b k (c h w) -> k b c h w', c=c, h=h, w=w)
        out2 = out2.sum(dim=0)

        k, b, c, h, w = x[2].shape

        x[2] = rearrange(x[2], 'k b c h w -> (k b) c h w')
        k3 = self.proj_in3(x[2])   # [(k b), c, h, w] = [(k b), 256, 32, 32]

        Q3 = rearrange(query[2], 'b c h w -> b 1 (c h w)')  # [b, c, h, w] = [(k b), 256, 32, 32]
        K3 = rearrange(k3, '(k b) c h w -> b k (c h w)', b=b, k=k)
        V3 = rearrange(x[2], '(k b) c h w -> b k (c h w)', b=b, k=k)   # [batch_size, h*w, c] = [3, 262144, 512]

        # [batch_size, h*w, seq_len]
        att_weights3 = torch.bmm(K3, Q3.transpose(1, 2))  # [b, k, 1]
        att_weights3 = att_weights3 * self.scale3

        att_weights3 = F.softmax(att_weights3, dim=1)
        out3 = V3 * att_weights3  # [b, k, (c h w)]
        out3 = rearrange(out3, 'b k (c h w) -> k b c h w', c=c, h=h, w=w)
        out3 = out3.sum(dim=0)


        return [out1, out2, out3]
