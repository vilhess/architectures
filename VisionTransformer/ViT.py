import torch
import torch.nn as nn


class CreatePatches(nn.Module):

    def __init__(self, in_channels, embedding_size, patch_size):
        super(CreatePatches, self).__init__()
        self.patch = nn.Conv2d(in_channels=in_channels, out_channels=embedding_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.patch(x).flatten(start_dim=2).transpose(1, 2)
    

class AttentionBlock(nn.Module):

    def __init__(self, embedding_size, hidden_dim, num_heads, dropout=0.0):
        super(AttentionBlock, self).__init__()
        self.pre_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.MLP = nn.Sequential(
            nn.Linear(embedding_size, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_size),
            nn.Dropout()
        )
    
    def forward(self, x):
        x_norm = self.pre_norm(x)
        x = x + self.attention(x_norm, x_norm, x_norm)[0]
        x_norm = self.norm(x)
        x = x + self.MLP(x_norm)
        return x
    

class ViT(nn.Module):

    def __init__(self, img_size=224, in_channel=3, embed_size=768, patch_size=16, hidden_dim=3072, num_heads=12, num_layers=12, num_classes=1000, dropout=0.0):
        super(ViT, self).__init__()

        self.patch_size = patch_size
        num_patches = (img_size//patch_size)**2
        self.patches = CreatePatches(in_channels=in_channel, embedding_size=embed_size ,patch_size=patch_size)

        # Positional encoding
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches+1, embed_size)) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

        self.att_layer = nn.ModuleList([])
        for _ in range(num_layers):
            self.att_layer.append(
                AttentionBlock(embedding_size=embed_size, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_size, eps=1e-6)
        self.head = nn.Linear(embed_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

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
        b, n, _ = x.shape

        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embeddings

        x = self.dropout(x)

        for layer in self.att_layer:
            x = layer(x)
        x = self.ln(x)
        x = x[:,0]
        x = self.head(x)
        return self.softmax(x)
    
if __name__=='__main__':
    
    model = ViT()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    x = torch.rand(10, 3, 224, 224)
    print(model(x).shape)