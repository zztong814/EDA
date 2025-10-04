import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- 工具函数 -----------------------------

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ----------------------------- Positional Encoding -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ----------------------------- Multi-Head Attention -----------------------------

class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == True, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# ----------------------------- Feed-Forward -----------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ----------------------------- 编码器层与模块 -----------------------------

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ----------------------------- Multi-input Transformer -----------------------------

class MultiInputTransformer(nn.Module):
    """
    接口: model(input, craft, taskid)
    - input:  [batch, 1, 13]    (连续特征)
    - craft:  [batch, 1, 1]     (数值)
    - taskid: [batch, 1, 2]     (两位二进制，float 或 long->one-hot)
    输出:
    - [batch, 13]  （对应 input 的预测）
    """
    def __init__(self, d_model=128, N=4, d_ff=512, h=4, dropout=0.1):
        super().__init__()
        # 将三部分分别投影到 d_model
        self.input_proj = nn.Linear(13, d_model)
        self.craft_proj = nn.Linear(1, d_model)
        self.task_proj = nn.Linear(2, d_model)

        # 不启用位置编码
        # self.pos_enc = PositionalEncoding(d_model, max_len=3, dropout=dropout)

        # 构造 Encoder
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(d_model, attn, ff, dropout)
        self.encoder = Encoder(encoder_layer, N)

        # 输出层：把第一个 token 映射回 13 维
        self.output = nn.Linear(d_model, 13)

        # 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_vec, craft, taskid, mask=None):
        """
        input_vec: [batch, 1, 13]
        craft:     [batch, 1, 1]
        taskid:    [batch, 1, 2]
        """
        # 投影
        i = self.input_proj(input_vec)   # [batch, 1, d_model]
        c = self.craft_proj(craft)       # [batch, 1, d_model]
        t = self.task_proj(taskid)       # [batch, 1, d_model]

        # 拼接成 seq_len = 3
        x = torch.cat([i, c, t], dim=1)  # [batch, 3, d_model]

        # 如果使用位置编码，则启用下面一行
        # x = self.pos_enc(x)

        # Encoder
        memory = self.encoder(x, mask)   # [batch, 3, d_model]

        # 取第一个 token（对应 input）的表示作为输出
        out_vec = self.output(memory[:, 0, :])  # [batch, 13]
        return out_vec


# ----------------------------- 测试脚本 -----------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiInputTransformer(d_model=128, N=3, d_ff=512, h=4, dropout=0.1).to(device)
    model.eval()

    # batch 示例（batch=1）
    batch = 1
    input_vec = torch.randn(batch, 1, 13, device=device)          # 实数特征
    craft = torch.randn(batch, 1, 1, device=device)               # 实数工艺参数
    # taskid 用二进制表示，例如 '01' -> [0,1]
    taskid = torch.tensor([[[0., 1.]]], device=device)            # shape [1,1,2]

    out = model(input_vec, craft, taskid)
    print("output shape:", out.shape)     # expected: [1, 13]
    print("output:", out.detach().cpu().numpy())
