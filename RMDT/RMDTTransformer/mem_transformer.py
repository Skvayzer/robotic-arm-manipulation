import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# sys.path.append('utils')
# from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
# from utils.log_uniform_sampler import LogUniformSampler, sample_logits

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None].bool(), -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None].bool(), -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 4:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,:].bool(), -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

# learnable token
class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None].bool(), -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None].bool(), -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.n_emb_projs = 0
        # parameter list is not supported by DataParallel
        # move all parameters from ParameterList to module attributes
        # self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                setattr(self, 'emb_projs_0', nn.Parameter(torch.Tensor(d_proj, d_embed)))
                self.n_emb_projs += 1
                # self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                # self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))
                setattr(self, f'emb_projs_{i}', nn.Parameter(torch.Tensor(d_proj, d_emb_i)))
                self.n_emb_projs += 1
            

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            inp_flat = inp.contiguous().view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                                    dtype=torch.float, 
                                    device=inp.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                # emb_i = F.linear(emb_i, self.emb_projs[i])
                emb_i = F.linear(emb_i, getattr(self, f'emb_projs_{i}'))

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed

class MemTransformerLM(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None, 
                 num_mem_tokens=None, read_mem_from_cache=False, mem_at_end=True,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, max_ep_len=100,
                 sample_softmax=-1, STATE_DIM=None, ACTION_DIM=None, IMG_DIM=None, n_token=None, DEP_DIM=None):
        super(MemTransformerLM, self).__init__()
        #self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.STATE_DIM = STATE_DIM
        self.ACTION_DIM = ACTION_DIM
        self.IMG_DIM = IMG_DIM
        self.mujoco_mode = None
        self.toto_mode = IMG_DIM is not None
        if self.mujoco_mode:
            # Mujoco mode
            self.embed_timestep = nn.Embedding(max_ep_len, d_embed)
            self.ret_emb = nn.Linear(1, d_embed)
            self.state_encoder = nn.Linear(self.STATE_DIM, d_embed) #RMT MUJOCO
            self.action_embeddings = nn.Linear(self.ACTION_DIM, d_embed) #RMT MUJOCO

            self.embed_ln = nn.LayerNorm(d_embed)


            self.head = nn.Sequential(*([nn.Linear(d_embed, self.ACTION_DIM)] + ([nn.Tanh()])))

        elif self.toto_mode:
            self.embed_timestep = nn.Embedding(max_ep_len, d_embed)
            self.ret_emb = nn.Linear(1, d_embed)
            self.state_encoder = nn.Linear(self.STATE_DIM, d_embed) #RMT MUJOCO
            self.action_embeddings = nn.Linear(self.ACTION_DIM, d_embed) #RMT MUJOCO
            self.img_embeddings = nn.Linear(self.IMG_DIM, d_embed) #RMT MUJOCO
            if DEP_DIM is not None:
                self.img_depth_embeddings = nn.Linear(DEP_DIM, d_embed)
            self.embed_ln = nn.LayerNorm(d_embed)
            self.head = nn.Sequential(*([nn.Linear(d_embed, self.ACTION_DIM)] + ([nn.Tanh()])))
        
        else:
            self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(3136, d_embed), nn.Tanh())

            self.ret_emb = nn.Sequential(nn.Linear(1, d_embed), nn.Tanh())

            self.action_embeddings = nn.Sequential(nn.Embedding(n_token, d_embed), nn.Tanh())
            self.head = nn.Linear(d_embed, n_token, bias=False)
        
        
        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = None
        self.num_mem_tokens = num_mem_tokens
        self.init_mem_tokens()
        self.read_mem_from_cache = read_mem_from_cache
        self.mem_at_end = mem_at_end

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0: # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        # self.sample_softmax = sample_softmax
        # # use sampled softmax
        # if sample_softmax > 0:
        #     self.out_layer = nn.Linear(d_model, n_token)
        #     if tie_weight:
        #         self.out_layer.weight = self.word_emb.weight
        #     self.tie_weight = tie_weight
        #     self.sampler = LogUniformSampler(n_token, sample_softmax)

        # # use adaptive softmax (including standard softmax)
        # else:
        #     self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
        #                                             cutoffs, div_val=div_val)

        #     if tie_weight:
        #         for i in range(len(self.crit.out_layers)):
        #             self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

        #     if tie_projs:
        #         for i, tie_proj in enumerate(tie_projs):
        #             if tie_proj and div_val == 1 and d_model != d_embed:
        #                 self.crit.out_projs[i] = self.word_emb.emb_projs[0]
        #                 setattr(self.crit, f'out_projs_{i}', self.word_emb.emb_projs_0)
        #             elif tie_proj and div_val != 1:
        #                 setattr(self.crit, f'out_projs_{i}', getattr(self.word_emb, f'emb_projs_{i}'))
        #                 self.crit.out_projs[i] = self.word_emb.emb_projs[i]
                        # self.crit.out_projs[i] = getattr(self.word_emb, f'emb_projs_{i}')

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.rand(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.rand(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, device):
        if self.mem_len > 0:
            mems = []
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=torch.float, device=device)
                mems.append(empty)

            return mems
        else:
            return None

    def init_mem_tokens(self):
        if self.num_mem_tokens == 0:
            self.mem_tokens = None
        else:
            mem_tokens = [torch.randn(1, self.d_model)] * self.num_mem_tokens
            mem_tokens = torch.cat(mem_tokens, dim=0).view(self.num_mem_tokens, 1, -1)
            mem_tokens = torch.nn.Parameter(mem_tokens, requires_grad=True)
            self.register_parameter(param=mem_tokens, name='mem_tokens')

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, word_emb, mems=None, mem_tokens=None, lengths=None):

        #word_emb = self.word_emb(dec_inp)
        bsz, _, _ = word_emb.size()
        word_emb = word_emb.permute(1,0,2)

        mlen = mems[0].size(0) if mems is not None else 0
        
        # Concat with mem_tokens
        if mem_tokens is not None:
            #print(mem_tokens.shape, word_emb.shape)
            #print(mem_tokens.shape, word_emb.shape, " Shapes here")
            word_emb = torch.cat((mem_tokens, word_emb), dim=0)
            #print(word_emb.shape)
            if self.mem_at_end:
                word_emb = torch.cat((word_emb, mem_tokens), dim=0)

        # qlen, bsz = dec_inp.size()
        #qlen = word_emb.shape[0]
        #klen = mlen + qlen
        qlen = word_emb.shape[0]
        klen = mlen + qlen
        
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()

            if self.num_mem_tokens != 0:
                dec_attn_mask[:self.num_mem_tokens, mlen:mlen+self.num_mem_tokens] = 0
                dec_attn_mask[:self.num_mem_tokens, :mlen] = 1 - int(self.read_mem_from_cache)
                if self.mem_at_end:
                    dec_attn_mask[-self.num_mem_tokens:, -self.num_mem_tokens:] = 0
                    dec_attn_mask[-self.num_mem_tokens:, :mlen] = 1 - int(self.read_mem_from_cache)
            dec_attn_mask = dec_attn_mask[:,:,None]
            if lengths is not None:
                dec_attn_mask = dec_attn_mask.unsqueeze(2).repeat((1, 1, bsz, 1))
                for i, length in enumerate(lengths):
                    dec_attn_mask[
                        length+self.num_mem_tokens:-self.num_mem_tokens, 
                        length+self.num_mem_tokens:-self.num_mem_tokens, i, :
                        ] = 1
            
        hids = []
        if self.attn_type == 0: # default
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                #print(core_out.shape, pos_emb.shape, dec_attn_mask.shape)
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hids, mems, qlen, mlen)

        return core_out, new_mems

    def forward(self, states, actions, rtgs, target, timesteps, *mems, mem_tokens=None, img=None, length=None, depth=None): 
        if self.mujoco_mode:
            if not mems: mems = self.init_mems(states.device)
            state_embeddings = self.state_encoder(states) 
            if actions is not None:
                rtg_embeddings = self.ret_emb(rtgs)
                action_embeddings = self.action_embeddings(actions)
                token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                time_embeddings = self.embed_timestep(timesteps)

                token_embeddings[:, ::3, :] = rtg_embeddings + time_embeddings
                token_embeddings[:, 1::3, :] = state_embeddings + time_embeddings
                token_embeddings[:, 2::3, :] = action_embeddings[:,-states.shape[1] + int(target is None):,:] + time_embeddings[:,-states.shape[1] + int(target is None):,:]
            else:
                rtg_embeddings = self.ret_emb(rtgs)

                token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
                token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]

            hidden, new_mems = self._forward(token_embeddings, mems=mems, mem_tokens=mem_tokens, q_len=length)
            hidden = hidden.permute(1,0,2)

            num_mem = self.num_mem_tokens

            if self.num_mem_tokens > 0:
                if self.mem_at_end:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -num_mem:, :]
                else:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -tgt_len-num_mem:-tgt_len, :]

            if self.mem_at_end:
                logits = self.head(hidden)[:, num_mem:-num_mem]
            else:
                tgt_len = token_embeddings.shape[1]
                logits = self.head(hidden)[:, -tgt_len:] # was tgt_len

            if actions is not None:
                logits = logits[:, 1::3, :]
            else:
                logits = logits[:, 1:, :]


            loss = None
            if target is not None:
                loss = nn.MSELoss()(logits, target)

            output = [logits, loss]
            if new_mems is not None:
                output = output + new_mems

            if self.num_mem_tokens != 0:
                output = output, mem_tokens_write.permute(1,0,2)
            else:
                output = output, None

            return output

        elif self.toto_mode:
            if not mems: mems = self.init_mems(states.device)
            if depth is not None and img is not None:
                num = 5
                img_embeddings = self.img_embeddings(img)
                depth_embeddings = self.img_depth_embeddings(depth)
            elif img is not None:
                num = 4
                img_embeddings = self.img_embeddings(img)
            elif depth is not None:
                num = 4
                depth_embeddings = self.img_depth_embeddings(depth)
            else:
                num = 3

            state_embeddings = self.state_encoder(states) 
            rtg_embeddings = self.ret_emb(rtgs)
            action_embeddings = self.action_embeddings(actions)
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*num - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
            timesteps = torch.arange(states.shape[1], device=token_embeddings.device)
            time_embeddings = self.embed_timestep(timesteps)
            time_embeddings = time_embeddings.unsqueeze(0).expand(states.shape[0], time_embeddings.shape[0], time_embeddings.shape[1])

            token_embeddings[:, ::num, :] = rtg_embeddings  + time_embeddings
            token_embeddings[:, 1::num, :] = state_embeddings  + time_embeddings
            token_embeddings[:, num-1::num, :] = action_embeddings[:,-states.shape[1] + int(target is None):,:] + time_embeddings[:,-states.shape[1] + int(target is None):,:]
            
            if depth is not None and img is not None:
                token_embeddings[:, 2::num, :] = img_embeddings  + time_embeddings
                token_embeddings[:, 3::num, :] = depth_embeddings  + time_embeddings
            elif img is not None:
                token_embeddings[:, 2::num, :] = img_embeddings  + time_embeddings
            elif depth is not None:
                token_embeddings[:, 2::num, :] = depth_embeddings  + time_embeddings
            
            hidden, new_mems = self._forward(token_embeddings, mems=mems, mem_tokens=mem_tokens, lengths=length)
            hidden = hidden.permute(1,0,2)

            num_mem = self.num_mem_tokens

            if self.num_mem_tokens > 0:
                if self.mem_at_end:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -num_mem:, :]
                else:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -tgt_len-num_mem:-tgt_len, :]

            if self.mem_at_end:
                logits = self.head(hidden)[:, num_mem:-num_mem]
            else:
                tgt_len = token_embeddings.shape[1]
                logits = self.head(hidden)[:, -tgt_len:] # was tgt_len

            if actions is not None:
                logits = logits[:, num-2::num, :]
            else:
                logits = logits[:, num-2:, :]


            loss = None
            if target is not None:
                loss = nn.MSELoss()(logits, target)

            output = [logits, loss]
            if new_mems is not None:
                output = output + new_mems

            if self.num_mem_tokens != 0:
                output = output, mem_tokens_write.permute(1,0,2)
            else:
                output = output, None

            return output
            
        else:
            if not mems: mems = self.init_mems(states.device)
            state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) 
            state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.d_embed) # (batch, block_size, n_embd)

            if actions is not None:
                rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
                action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
                token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(target is None), self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:, ::3, :] = rtg_embeddings
                token_embeddings[:, 1::3, :] = state_embeddings
                token_embeddings[:, 2::3, :] = action_embeddings[:,-states.shape[1] + int(target is None):,:]

            else:
                rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

                token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.d_embed), dtype=torch.float32, device=state_embeddings.device)
                token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
                token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]




            hidden, new_mems = self._forward(token_embeddings, mems=mems, mem_tokens=mem_tokens)
            hidden = hidden.permute(1,0,2)



            num_mem = self.num_mem_tokens
            if self.num_mem_tokens > 0:
                if self.mem_at_end:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -num_mem:, :]
                else:
                    tgt_len = token_embeddings.shape[1]
                    mem_tokens_write = hidden[:, -tgt_len-num_mem:-tgt_len, :]
            if self.mem_at_end:
                logits = self.head(hidden)[:, num_mem:-num_mem]
            else:
                tgt_len = token_embeddings.shape[1]
                logits = self.head(hidden)[:, -tgt_len:] # was tgt_len


            if actions is not None:
                logits = logits[:, 1::3, :]
            else:
                logits = logits[:, 1:, :]



            loss = None
            if target is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            output = [logits, loss]
            if new_mems is not None:
                output = output + new_mems

            if self.num_mem_tokens != 0:
                output = output, mem_tokens_write.permute(1,0,2)
            else: 
                output = output, None

            return output