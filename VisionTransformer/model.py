import torch
import torch.nn as nn


class CreatePatches(nn.Module):
    def __init__(self, in_channels, embedding_size, patch_size):
        super(CreatePatches, self).__init__()
        self.patch = nn.Conv2d(in_channels=in_channels, out_channels=embedding_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.patch(x).flatten(start_dim=2).transpose(1, 2)
    

class AttentionBlock(nn.Module):
    def __init__(self, embeddings_size, hidden_dim, num_heads, dropout=0.0):
        super(AttentionBlock, self).__init__()

        self.prenorm = nn.LayerNorm(embeddings_size)
        self.mha = nn.MultiheadAttention(embed_dim=embeddings_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embeddings_size)
        self.mlp = nn.Sequential(
            nn.Linear(embeddings_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embeddings_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.prenorm(x)
        x_norm = self.mha(x_norm, x_norm, x_norm)[0]
        x = x + x_norm
        x_norm = self.norm(x)
        x_norm = self.mlp(x_norm)
        return x + x_norm
    

class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, in_channels=1, embed_size=768, patch_size=4, hidden_dim=256, num_heads=12, num_layers=12, num_classes=10, dropout=0.0):
        super(VisionTransformer, self).__init__()

        num_patches = (img_size//patch_size)**2

        self.patches = CreatePatches(in_channels, embed_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches+1, embed_size))

        self.atts = nn.ModuleList([])
        for _ in range(num_layers):
            self.atts.append(AttentionBlock(embed_size, hidden_dim, num_heads))
        
        self.head = nn.Linear(embed_size, num_classes)
        self.ln = nn.LayerNorm(embed_size)
        self.softmax = nn.Softmax(dim=1)
        self.dp = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patches(x)
        batch, num_patches, embed_size = x.shape
        cls_token = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((x, cls_token), dim=1)
        x += self.pos_embeddings
        x = self.dp(x)

        for layer in self.atts:
            x = layer(x)
        x = self.ln(x)
        x = x[:,0]
        x = self.head(x)
        return self.softmax(x)
    

if __name__ == '__main__':
    img = torch.rand(12, 1, 28, 28)

    vit = VisionTransformer()
    print(vit(img).shape)
