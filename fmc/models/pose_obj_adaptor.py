import math
import torch
import torch.nn as nn
from einops import rearrange


class CamObjPoseAdaptor(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        self.pose_encoder = pose_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, pose_embedding,traj_features):
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]            # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)      # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs)
                                   for x in pose_embedding_features]
        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               pose_embedding_features=pose_embedding_features,traj_features=traj_features).sample
        return noise_pred
