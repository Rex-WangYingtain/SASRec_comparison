import torch
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from  performer_pytorch import *


class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, local_heads=0, attn_out_bias=True, dropout=0.):
        super(MutiHeadAttention, self).__init__()
        # print(d_model)    50
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head

        self.global_heads = n_head - local_heads

        self.dim_heads = self.d_model // self.n_head

        # q，k，v的权重矩阵
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fast_attention = FastAttention(self.dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False)

        self.to_out = nn.Linear(d_model, d_model, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)



    def forward(self, q, k, v, mask=None, context_mask=None):
        gh = self.global_heads
        h = self.n_head

        cross_attend = exists(k)
        # 线性变换
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            # if exists(pos_emb) and not cross_attend:
            #     q, k = apply_rotary_pos_emb(q, k, pos_emb)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)


if __name__=='__main__':
    d_model = 512
    n_head = 8

    # batch time dimension
    x = torch.rand(128, 32, d_model)

    mask = [False for _ in range(64)] + [True for _ in range(64)]
    mask = Tensor(mask).type(dtype=torch.bool)

    attention = MutiHeadAttention(d_model=d_model, n_head=n_head, local_heads=0, attn_out_bias=True, dropout=0.)

    print(x.shape)

    out = attention(x, x, x, mask=mask, context_mask=None)
    print(out)
