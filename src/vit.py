import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class PatchEmbedding(nn.Layer):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2D(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 输入: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = paddle.flatten(x, 2).transpose([0, 2, 1])  # [B, num_patches, embed_dim]
        return x

class Attention(nn.Layer):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = paddle.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, C//num_heads]
        
        # 矩阵乘法
        attn = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        
        x = paddle.matmul(attn, v)
        x = paddle.transpose(x, [0, 2, 1, 3])
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Layer):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoderLayer(nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=100, embed_dim=192,
                 depth=6, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = self.create_parameter(shape=[1, 1, embed_dim], default_initializer=nn.initializer.Constant(0.0))
        self.pos_embed = self.create_parameter(shape=[1, (img_size // patch_size) ** 2 + 1, embed_dim], default_initializer=nn.initializer.Constant(0.0))
        self.pos_drop = nn.Dropout(p=dropout)

        self.layers = nn.LayerList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = paddle.expand(self.cls_token, [B, -1, -1])  # [B, 1, embed_dim]
        x = paddle.concat([cls_tokens, x], axis=1)  # [B, num_patches+1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        cls_logits = self.head(x[:, 0])
        return cls_logits

if __name__ == '__main__':
    paddle.set_device('gpu')
    model = VisionTransformer(
        img_size=32,        # 调整图像大小
        patch_size=8,       # Patch 大小
        in_channels=3,       # 输入通道数
        num_classes=100,     # CIFAR-100 类别数
        embed_dim=64,       # 嵌入维度
        depth=6,             # Transformer 层数
        num_heads=4,         # 多头数量
        mlp_ratio=4.0,       # MLP 隐藏层比例
        dropout=0.1          # Dropout 概率
    )
    x = paddle.randn([128, 3, 32, 32])
    y = model(x)
    print(f"Model output shape: {y.shape}")
