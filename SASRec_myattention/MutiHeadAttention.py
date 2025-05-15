import torch
from torch import nn
import math

class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MutiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head

        # q，k，v的权重矩阵
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # q，k，v计算合到一起后的权重矩阵
        self.w_combine = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        # 每个头的维度
        n_d = self.d_model // self.n_head
        # 线性变换
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 将q，k，v重塑为多头的形状？
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        # 计算注意力分数
        score = q@k.transpose(2, 3) / math.sqrt(n_d)
        # 考虑掩码的情况
        if mask is not None:
            score = score.masked_fill(mask==0, -10000)
        score = self.softmax(score)@v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        out = self.w_combine(score)
        return out


if __name__=='__main__':
    d_model = 512
    n_head = 8

    # batch time dimension
    x = torch.rand(128, 32, d_model)

    attention = MutiHeadAttention(d_model=d_model, n_head=n_head)
    out = attention(x, x, x)
    print(out)
