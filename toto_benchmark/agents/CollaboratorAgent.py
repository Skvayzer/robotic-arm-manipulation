"""
If you are contributing a new agent, implement your 
agent initialization & predict functions here. 

Next, (optionally) update your agent's name in agent/__init__.py.
"""

import numpy as np
import torchvision.transforms as T
from .Agent import Agent
from RMDT.RMDTTransformer import mem_transformer
import torch
import os
from toto_benchmark.vision import load_model, load_transforms
from toto_benchmark.vision.Resnet import _load_model

NUM_JOINTS = 7

class CollaboratorAgent(Agent):
    def __init__(
            self, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt, MEM_LEN, ext_len, 
            num_mem_tokens, mem_at_end, device, learning_rate, context_length, sections, grad_norm_clip,
            img_depth_encoder=None,):
        
        self.model = mem_transformer.MemTransformerLM(n_layer=n_layer, n_head=n_head, 
                                                      d_model=d_model, d_head=d_head, d_inner=d_inner, 
                                                      dropout=dropout, dropatt=dropatt, mem_len=MEM_LEN, 
                                                      ext_len=ext_len, num_mem_tokens=num_mem_tokens, max_ep_len=context_length,
                                                      mem_at_end=mem_at_end, STATE_DIM=7, ACTION_DIM=7, IMG_DIM=2048, DEP_DIM=128)
        self.img_depth_encoder = img_depth_encoder
        if self.img_depth_encoder:
            self.img_depth_encoder.to(device)
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.EFFECTIVE_SIZE_BLOCKS = context_length * sections
        self.BLOCKS_CONTEXT = context_length
        self.learning_rate = learning_rate
        self.grad_norm_clip = grad_norm_clip
        self.buffer = {
            'observations': None, 
            'actions': None, 
            'rewards': None, 
            'embeddings': None,
            'embeddings_img_depth': None,
        }
        self.reward_init = torch.tensor([[121.91]], device=device)
        self.memory = None
        self.mem_tokens=None
        self.training = False
        self.predict_actions = False

    def forward(self, batch):
        length = batch["length"].detach().cpu().numpy() if "length" in batch else None
        lengths = None

        if batch["observations"].shape[1] > self.BLOCKS_CONTEXT:
            for block_part in range(self.EFFECTIVE_SIZE_BLOCKS//self.BLOCKS_CONTEXT):
                from_idx = block_part*(self.BLOCKS_CONTEXT)
                to_idx = min((block_part+1)*(self.BLOCKS_CONTEXT), batch["observations"].shape[1])
                x1 = batch["observations"][:, from_idx:to_idx, :].to(self.device)
                y1 = batch["actions"][:, from_idx:to_idx, :].to(self.device)
                r1 = batch["rewards"][:, from_idx:to_idx, :].to(self.device)
                i1 = None
                d1 = None
                if "embeddings" in batch:
                    i1 = batch["embeddings"][:, from_idx:to_idx, :].to(self.device)
                if self.img_depth_encoder and "cam0d" in batch:
                    inp = batch["cam0d"][:, from_idx:to_idx, :]
                    bsz, horizon, img_h, img_w = list(inp.shape)
                    inp = inp.reshape(-1, img_h, img_w)
                    inp = inp.unsqueeze(1)
                    d1 = self.img_depth_encoder(inp).reshape((bsz, horizon, -1))
                    d1 = d1.to(self.device)
                if length is not None:
                    lengths = [l if l < self.BLOCKS_CONTEXT*4 else self.BLOCKS_CONTEXT*4 for l in length]
                    length = [max(0, (l - self.BLOCKS_CONTEXT*4)) for l in length]
                
                if self.mem_tokens is not None:
                    self.mem_tokens = self.mem_tokens.detach()
                elif self.raw_model.mem_tokens is not None:
                    self.mem_tokens = self.raw_model.mem_tokens.repeat(1, x1.shape[0], 1)
                        
                with torch.set_grad_enabled(self.training):
                    if self.memory is not None:
                        res = self.model(x1, y1, r1, y1 if not self.predict_actions else None, None, *self.memory, mem_tokens=self.mem_tokens, img=i1, length=lengths, depth=d1) # timesteps = None
                    else:
                        res = self.model(x1, y1, r1, y1 if not self.predict_actions else None, None, mem_tokens=self.mem_tokens, img=i1, length=lengths, depth=d1)
                    self.memory = res[0][2:]
                    logits, loss = res[0][0], res[0][1]
                    
                    self.mem_tokens = res[1]
                    
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                if length and sum(length) == 0:
                    break
                elif batch["observations"].shape[1] <= to_idx:
                    break
        else:
            x1 = batch["observations"].to(self.device)
            y1 = batch["actions"].to(self.device)
            r1 = batch["rewards"].to(self.device)
            i1 = None
            d1 = None
            if "embeddings" in batch:
                i1 = batch["embeddings"].to(self.device)
            if self.img_depth_encoder and "cam0d" in batch:
                inp = batch["cam0d"]
                bsz, horizon, img_h, img_w = list(inp.shape)
                inp = inp.reshape(-1, img_h, img_w)
                d1 = self.img_depth_encoder(inp).reshape((bsz, horizon, -1))
                d1 = d1.to(self.device)
            if self.mem_tokens is not None:
                self.mem_tokens = self.mem_tokens.detach()
            elif self.raw_model.mem_tokens is not None:
                self.mem_tokens = self.raw_model.mem_tokens.repeat(1, x1.shape[0], 1)
            if self.memory is not None:
                res = self.model(x1, y1, r1, y1 if not self.predict_actions else None, None, *self.memory, mem_tokens=self.mem_tokens, img=i1, length=lengths, depth=d1) # timesteps = None
            else:
                res = self.model(x1, y1, r1, y1 if not self.predict_actions else None, None, mem_tokens=self.mem_tokens, img=i1, length=lengths, depth=d1)
            self.memory = res[0][2:]
            logits, loss = res[0][0], res[0][1]
            self.mem_tokens = res[1]

        return logits, loss

    def predict(self, sample):
        self.model.eval()
        self.training = False
        self.predict_actions = True
        with torch.no_grad():
            self.pack_one_batch(sample)
            output = self.forward(self.buffer)[0]
            self.buffer["actions"][:, -1] = output
            return output.squeeze().to('cpu').detach().numpy()
    
    def pack_one_batch(self, sample):
        img = sample["cam0c"]
        imgs_out = self.img_encoder(torch.unsqueeze(img, dim=0).to(self.device))
        v = sample["inputs"]
        t = v if torch.is_tensor(v) else torch.from_numpy(v)
        t = t.float().unsqueeze(0).to(self.device)
        sample["observations"] = torch.clone(t)
        sample["actions"] = torch.clone(t)
        sample["embeddings"] = imgs_out
        sample["rewards"] = self.reward_init
        for key in self.buffer:
            if self.buffer[key] is None:
                self.buffer[key] = sample[key].unsqueeze(0)
            else:
                self.buffer[key] = torch.cat((self.buffer[key], sample[key].unsqueeze(0)), dim=1)[:, max(0, (self.buffer[key].shape[0] - self.EFFECTIVE_SIZE_BLOCKS)):]
    
    def train(self, batch):
        self.model.train()
        self.training = True
        self.memory = None
        self.mem_tokens=None
        logits, loss = self.forward(batch)
        self.model.zero_grad()
        if self.img_depth_encoder:
            self.img_depth_encoder.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        return loss
    
    def eval(self, batch):
        self.model.eval()
        self.training = False
        self.memory = None
        self.mem_tokens=None
        logits, loss = self.forward(batch)
        return loss.item()
    
    def save(self, foldername, filename='Agent.pth', epoch=0):
        state = {'epoch': epoch,
                 'optimizer': self.optimizer.state_dict(),
                 }
        state["model"] = self.model.state_dict()
        torch.save(state, os.path.join(foldername, filename))

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=torch.device(self.device))
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint["model"])


def _init_agent_from_config(config, device='cpu'):
    img_depth_encoder = _load_model(config)
    agent = CollaboratorAgent(
        n_layer=config.agent.n_layer, 
        n_head=config.agent.n_head, 
        d_model=config.agent.d_model, 
        d_head=config.agent.d_head, 
        d_inner=config.agent.d_inner, 
        dropout=config.agent.dropout, 
        dropatt=config.agent.dropatt, 
        MEM_LEN=config.agent.MEM_LEN, 
        ext_len=config.agent.ext_len, 
        num_mem_tokens=config.agent.num_mem_tokens, 
        mem_at_end=config.agent.mem_at_end,
        device=device,
        learning_rate=config.agent.learning_rate,
        context_length=config.agent.context_length,
        sections=config.agent.sections,
        grad_norm_clip=config.agent.grad_norm_clip,
        img_depth_encoder=img_depth_encoder,
        )
    if "agent_path" in config.agent:
        agent.load(config.agent.agent_path)
    transforms = lambda x: x
    return agent, transforms