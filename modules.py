
import torch
import torch.nn as nn
import yaml

class FilterLayer(nn.Module):
    def __init__(self, config):
        super(FilterLayer, self).__init__()
        # 初始化可训练的复杂权重参数
        self.complex_weight = nn.Parameter(torch.randn(1, config["max_seq_length"] // 2 + 1, config["hidden_size"], 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.LayerNorm = nn.LayerNorm(config["hidden_size"], eps=1e-12)

    def forward(self, input_tensor):
        # 输入张量形状 [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        # 对输入张量进行快速傅里叶变换（FFT）
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        # 将权重转换为复数权重
        weight = torch.view_as_complex(self.complex_weight)

        # 进行频域滤波
        x = x * weight

        # 进行逆快速傅里叶变换（iFFT）
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')

        # Dropout 处理
        hidden_states = self.out_dropout(sequence_emb_fft)

        # LayerNorm 处理并加上原始输入张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
