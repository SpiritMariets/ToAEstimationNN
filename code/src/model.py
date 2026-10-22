import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PositionwiseConvFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv2(self.dropout(self.act(self.conv1(x))))
        return x.transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = PositionwiseConvFFN(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.enc_dec_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = PositionwiseConvFFN(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        enc_dec_out, _ = self.enc_dec_attn(query=x, key=enc_out, value=enc_out)
        x = self.norm2(x + self.dropout(enc_dec_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class MPDelayTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        n_blocks: int = 4,
        n_heads: int = 4,
        d_m: int = 64,
        d_ff: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_m = d_m
        self.enc_input_conv = nn.Conv1d(in_channels, d_m, kernel_size=1)
        self.dec_input_conv = nn.Conv1d(1, d_m, kernel_size=1)
        self.pos_encoder = PositionalEncoding(d_m)
        self.pos_decoder = PositionalEncoding(d_m)
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_m, n_heads, d_ff, dropout) for _ in range(n_blocks)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_m, n_heads, d_ff, dropout) for _ in range(n_blocks)
        ])
        
        self.output_conv = nn.Conv1d(d_m, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        enc_x = self.enc_input_conv(src).transpose(1, 2)
        enc_x = self.pos_encoder(enc_x)
        for block in self.encoder_blocks:
            enc_x = block(enc_x)
            
        tgt_unsq = tgt.unsqueeze(1)
        dec_x = self.dec_input_conv(tgt_unsq).transpose(1, 2)
        dec_x = self.pos_decoder(dec_x)
        
        causal_mask = self.generate_causal_mask(tgt.size(1), tgt.device)
        for block in self.decoder_blocks:
            dec_x = block(dec_x, enc_x, tgt_mask=causal_mask)
            
        dec_x = dec_x.transpose(1, 2)
        out = self.output_conv(dec_x).squeeze(1)
        return self.sigmoid(out)

    @torch.no_grad()
    def predict(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        threshold: float = 0.95,
        scale_factor: float = 1.0
    ) -> torch.Tensor:
        self.eval()
        device = src.device
        
        enc_x = self.enc_input_conv(src).transpose(1, 2)
        enc_x = self.pos_encoder(enc_x)
        for block in self.encoder_blocks:
            enc_x = block(enc_x)
            
        tgt = torch.tensor([[0.0]], device=device)
        for _ in range(max_len):
            tgt_unsq = tgt.unsqueeze(1)
            dec_x = self.dec_input_conv(tgt_unsq).transpose(1, 2)
            dec_x = self.pos_decoder(dec_x)
            
            causal_mask = self.generate_causal_mask(tgt.size(1), device)
            for block in self.decoder_blocks:
                dec_x = block(dec_x, enc_x, tgt_mask=causal_mask)
                
            dec_x = dec_x.transpose(1, 2)
            out = self.sigmoid(self.output_conv(dec_x)).squeeze(1)
            
            next_val = out[:, -1:]
            if next_val.item() >= threshold:
                break
            tgt = torch.cat([tgt, next_val], dim=1)
            
        return tgt[0, 1:] * scale_factor


def check_model_weights(model: torch.nn.Module) -> bool:
    """Вспомогательный метод для проверки весов на NaN и Inf."""
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ Найдены NaN в слое: {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"❌ Найдены Inf в слое: {name}")
            has_nan = True
            
    if not has_nan:
        print("✅ Все веса модели корректны (NaN и Inf не обнаружены).")
    return has_nan
