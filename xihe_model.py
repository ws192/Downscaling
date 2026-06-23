import torch
import torch.nn as nn
import torch.nn.functional as F


class OceanSpecificTransformer(nn.Module):
    def __init__(self, input_channels=12, num_layers=12, patch_size=4, embed_dim=64, mask=None):
        super(OceanSpecificTransformer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask = mask  # Ocean-Land Mask

        # Patch Partition
        self.patch_partition = nn.Conv2d(input_channels, embed_dim, kernel_size=3, stride=1,padding=1,
                                         bias=False)

        # Transformer Layers (Ocean-Specific Blocks)
        self.transformer_layers = nn.ModuleList([
            OceanSpecificBlock(embed_dim, num_heads=4, mask=self.mask) for _ in range(num_layers)
        ])

        # Patch Restoration
        self.patch_restoration = nn.ConvTranspose2d(embed_dim, input_channels, kernel_size=patch_size,
                                                    stride=patch_size, bias=False)

    def forward(self, x):
        # Patch Partition
        print(x.shape)
        x = self.patch_partition(x)  # Shape: (B, embed_dim, H//patch_size, W//patch_size)
        print(x.shape)
        # Apply Ocean-Specific Transformer Blocks
        for layer in self.transformer_layers:
            x = layer(x)

        # Patch Restoration
        x = self.patch_restoration(x)  # Shape: (B, input_channels, H, W)
        return x


class OceanSpecificBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mask=None):
        super(OceanSpecificBlock, self).__init__()
        self.local_sie = LocalSIE(embed_dim, num_heads, mask)
        self.global_sie = GlobalSIE(embed_dim, num_heads)

    def forward(self, x):
        x = self.local_sie(x)
        x = self.global_sie(x)
        return x


class LocalSIE(nn.Module):
    def __init__(self, embed_dim, num_heads, mask=None):
        super(LocalSIE, self).__init__()
        self.mask = mask  # Ocean-land mask
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        # Input shape: (Batch, Channels, Height, Width)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # Reshape to (HW, B, C)

        # Apply Mask if provided
        if self.mask is not None:
            mask = self.mask.flatten(2).squeeze(1)  # Shape: (1, H*W)
            attention_mask = (mask == 0).type(torch.bool)  # Land is true (masked)
        else:
            attention_mask = None

        x = self.norm(x)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)  # Multi-head attention
        x += self.mlp(x)
        x = x.permute(1, 2, 0).view(B, C, H, W)  # Reshape back
        return x


class GlobalSIE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GlobalSIE, self).__init__()
        self.group_vectors = nn.Parameter(torch.randn(16, embed_dim))  # Example: 16 groups
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        # Input shape: (Batch, Channels, Height, Width)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # Reshape to (HW, B, C)

        # Global Group Attention
        group_vectors = self.group_vectors.unsqueeze(1).expand(-1, B, -1)  # Shape: (Groups, B, C)
        group_vectors, _ = self.attention(group_vectors, x, x)  # Global attention
        group_vectors = self.mlp(self.norm(group_vectors))

        # Propagate Global Info back to Patches
        x, _ = self.attention(x, group_vectors, group_vectors)
        x = x.permute(1, 2, 0).view(B, C, H, W)  # Reshape back
        return x


# Example Usage
if __name__ == '__main__':
    # Simulated Input (1, 12, 40, 40)
    input_tensor = torch.randn(1, 12, 40, 40)

    # Ocean-Land Mask (1, 1, 40, 40)
    ocean_land_mask = torch.ones(1, 1, 40, 40)
    ocean_land_mask[:, :, :10, :10] = 0  # Example: Mask out top-left corner

    # Model
    model = OceanSpecificTransformer(input_channels=12, mask=ocean_land_mask)
    output = model(input_tensor)
    print("Output shape:", output.shape)
