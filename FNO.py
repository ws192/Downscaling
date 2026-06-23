import torch
import torch.nn as nn
import torch.fft


# ======== FNO 模块：多尺度频域特征提取 ========
class FNO2D(nn.Module):
    def __init__(self, modes1, modes2, width):
        """
        Args:
            modes1: 傅里叶变换在第一维的模式数（频率范围）
            modes2: 傅里叶变换在第二维的模式数（频率范围）
            width: 嵌入维度
        """
        super(FNO2D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # 定义线性变换
        self.fc0 = nn.Linear(3, self.width)  # 例如输入是 (wind_u, wind_v, temp)
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.width)

        # 注意：FNO 使用频域加卷积
        self.fourier_weight = nn.Parameter(torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        """
        Args:
            x: 输入数据 (B, H, W, C)，例如网格点 (u, v, temp)
        Returns:
            输出特征 (B, H, W, C)，特征中包含频域信息
        """
        x = self.fc0(x)  # 输入维度提升到 width

        # 对通道进行离散傅里叶变换
        x_ft = torch.fft.rfft2(x, norm="forward")  # (B, H, W, width)，转到频域

        # 提取低频与高频模式
        x_ft = x_ft[..., :self.modes1, :self.modes2]
        x_ft = torch.einsum("bixy,ioxy->boxy", x_ft, self.fourier_weight)  # 点乘权重

        # 回到时域
        x = torch.fft.irfft2(x_ft, s=(x.size(-2), x.size(-1)), norm="forward")  # (B, H, W, width)

        # 后续处理
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ======== Transformer Block 模块 ========
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        改进版 Transformer，实现全局关系建模
        Args:
            d_model: 输入特征维度
            nhead: 多头注意力
            dim_feedforward: FFN的隐藏层维度
            dropout: Dropout 概率
        """
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, N, D)，如 (Batch, 网格点数, 特征数)
        Returns:
            输出特征 (B, N, D)
        """
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x)  # (B, N, D)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # FFN
        ffn_output = self.ffn(x)  # (B, N, D)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x


# ======== 改进版 TransformerBlock 整合 FNO ========
class FNOTransformerBlock(nn.Module):
    def __init__(self, modes1, modes2, width, d_model, nhead, dim_feedforward, dropout=0.1):
        """
        将 FNO 与 Transformer 结合，用于大气场特征提取
        Args:
            modes1, modes2: FNO 的频率模式范围
            width: FNO 的嵌入维度
            d_model: Transformer 输入特征维度
            nhead: Transformer 的多头注意力头数
            dim_feedforward: Transformer FFN 的隐藏层维度
            dropout: Transformer 的 Dropout 概率
        """
        super(FNOTransformerBlock, self).__init__()
        self.fno = FNO2D(modes1, modes2, width)
        self.transformer = TransformerBlock(d_model, nhead, dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(width, d_model)  # 调整 FNO 的输出到 Transformer 的输入特征维度

    def forward(self, x):
        """
        Args:
            x: 输入特征 (B, H, W, C)
        Returns:
            输出特征 (B, N, D)，如 (Batch, 网格点数, Transformer特征维度)
        """
        B, H, W, C = x.shape

        # FNO 提取多尺度特征
        x = self.fno(x)  # (B, H, W, width)

        # 展平为 Transformer 输入格式
        x = x.view(B, H * W, -1)  # (B, H*W, width)
        x = self.fc(x)  # (B, H*W, d_model)

        # Transformer 提取全局依赖
        x = self.transformer(x)  # (B, H*W, d_model)

        return x


# 模型实例化
model = FNOTransformerBlock(
    modes1=16, modes2=16, width=64,    # FNO 参数
    d_model=128, nhead=8,             # Transformer 参数
    dim_feedforward=256, dropout=0.1  # Transformer 参数
)

# 假设输入形状：(Batch, Height, Width, Channels)
x = torch.randn(8, 64, 64, 3)  # 例如 8 个样本，64x64 网格，3 个物理变量

# 运行模型
output = model(x)  # (8, 64*64, 128)
print("输出特征形状：", output.shape)
