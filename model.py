import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self,features:int,eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.ones(features))

    def forward(self,x):
            mean=x.mean(dim=-1,keepdim=True)
            std=x.std(dim=-1,keepdim=True)
            return self.alpha*(x-mean)/(std+self.eps)+self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ff,d_model)

    def forward(self,x):
            return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class InputEmbedding(nn.Module):

    def __init__(self,d_model:int, vocab_size:int)->None:
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        pe=torch.zeros(seq_len,d_model)
        position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)

        div_term=torch.exp(torch.arange(0,d_model ,2).float()*(-math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)

        pe[:, 1::2]=torch.cos(position*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+(self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features:int ,dropout:float)->None:
        super().__init__()
        self.norm=LayerNormalization(features)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class MultiHeadAttention(nn.Module):

    def __init__(self,d_model:int,h:int,dropout:float)->None:
        super().__init__()
        self.d_model=d_model
        self.h=h

        assert d_model%h==0

        self.d_k=d_model // h
        self.w_k=nn.Linear(d_model,d_model,bias=False)
        self.w_v=nn.Linear(d_model,d_model,bias=False)
        self.w_q=nn.Linear(d_model,d_model,bias=False)
        self.w_o=nn.Linear(d_model,d_model,bias=False)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attention_score=(query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_score.masked_fill(mask==0,-1e9)

        attention_score=attention_score.softmax(dim=-1)

        if dropout is not None:
            attention_score=dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self,q,k,v,mask):
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x,self.attention_scores=MultiHeadAttention.attention(query,key,value,mask,self.dropout)

        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k)

        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
        return x

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def  __init__(self , features:int ,self_attention_block:MultiHeadAttention, feed_forward_block:FeedForwardBlock , cross_attention_block:MultiHeadAttention, dropout:float)->None:
        super().__init__()
        self.self_attention_block=self_attention_block
        self.feed_forward_block=feed_forward_block
        self.cross_attention_block=cross_attention_block
        self.residual_connection=nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connection[0](x,lambda x : self.self_attention_block(x,x,x,tgt_mask))
        x=self.residual_connection[1](x,lambda x : self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection[2](x,lambda x : self.feed_forward_block(x))
        return x

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int)->None:
        super().__init__()
        self.proj=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        return self.proj(x)

class Transformer(nn.Module):

    def __init__(self,encoder:Encoder , decoder:Decoder ,src_embed:InputEmbedding , tgt_embed:InputEmbedding , Projection_layer:ProjectionLayer,src_pos:PositionalEncoding,tgt_pos:PositionalEncoding)->None:
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.Projection_layer=Projection_layer
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos

    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output:torch.Tensor ,src_mask:torch.Tensor ,tgt:torch.Tensor,tgt_mask:torch.Tensor):
        tgt=self.tgt_embed(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)

    def project(self,x):
        return self.Projection_layer(x)


def build_transformer(src_vocab_size : int ,tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model:int=512,N:int=6,h:int=8,d_ff:int=2048,dropout:float=0.1)->Transformer:

    src_embed=InputEmbedding(d_model,src_vocab_size)
    tgt_embed=InputEmbedding(d_model,tgt_vocab_size)


    src_pos=PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos=PositionalEncoding(d_model,tgt_seq_len,dropout)

    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention=MultiHeadAttention(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=EncoderBlock(d_model,encoder_self_attention,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for _ in range(N):
        decoder_self_attention_block=MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention_block=MultiHeadAttention(d_model,h,dropout)
        feed_forward_block=FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(d_model,decoder_self_attention_block,feed_forward_block,decoder_cross_attention_block,dropout)
        decoder_blocks.append(decoder_block)

    encoder=Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder=Decoder(d_model,nn.ModuleList(decoder_blocks))

    Projection_layer=ProjectionLayer(d_model,tgt_vocab_size)

    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,Projection_layer,src_pos,tgt_pos)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer