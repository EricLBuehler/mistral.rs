# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, PretrainedConfig
from transformers import CLIPVisionConfig 
from transformers.utils import logging
from datetime import datetime 

logger = logging.get_logger(__name__)

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
  attention_dropout=0.0,
  dropout=0.0,
  hidden_act="quick_gelu",
  hidden_size=1024,
  image_size=336,
  initializer_factor=1.0,
  initializer_range=0.02,
  intermediate_size=4096,
  layer_norm_eps=1e-05,
  num_attention_heads=16,
  num_channels=3,
  num_hidden_layers=24,
  patch_size=14,
  projection_dim=768 
)

class Phi3ImageEmbedding(nn.Module):
    """Phi3 Image embedding."""

    def __init__(self, config: PretrainedConfig, wte=None, **kwargs) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        self.wte = wte

        if isinstance(config.img_processor, dict) and config.img_processor.get('name', None) == 'clip_vision_model':
            assert 'model_name' in config.img_processor, 'model_name must be provided for CLIPVisionModel'
            assert 'image_dim_out' in config.img_processor, 'image_dim_out must be provided for CLIPVisionModel'
            assert 'num_img_tokens' in config.img_processor, 'num_img_tokens must be provided for CLIPVisionModel'
            assert config.img_processor['model_name'] == 'openai/clip-vit-large-patch14-336'
            clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
            self.img_processor = CLIPVisionModel(clip_config)
            image_dim_out = config.img_processor['image_dim_out']
            self.num_img_tokens = config.img_processor['num_img_tokens']
        else:
            raise NotImplementedError(f'img_processor = {config.img_processor}, not implemented')

        self.image_dim_out = image_dim_out
        self.img_sizes = None

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = kwargs.get('use_hd_transform', False)
        self.with_learnable_separator = kwargs.get('with_learnable_separator', False)
        self.hd_transform_order = kwargs.get('hd_transform_order', 'glb_sub')
        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform == self.with_learnable_separator, 'use_hd_transform and with_learnable_separator should have same value'
        if self.with_learnable_separator:
            assert self.use_hd_transform, 'learnable separator is only for hd transform'
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * 4]))
            self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * 4]))
            logger.info(f'learnable separator enabled for hd transform, hd_transform_order = {self.hd_transform_order}')

        projection_cls = kwargs.get('projection_cls', 'linear')
        if projection_cls == 'linear':
            self.img_projection = nn.Linear(image_dim_out, hidden_size)
        elif projection_cls == 'mlp' and self.use_hd_transform:
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out * 4, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        elif projection_cls == 'mlp':
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')
        self.layers = layers
        self.vocab_size = config.vocab_size
        self.img_features = None

        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get('type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'


    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        self.img_sizes = img_sizes

    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature
        img_processor_output = self.img_processor(torch.full_like(img_embeds, 0.5), output_hidden_states=True)
        img_feature = img_processor_output.hidden_states[LAYER_IDX]
        img_feature = torch.ones_like(img_feature)
        

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == "cls_patch":
            return img_feature

        raise NotImplementedError

    def forward(self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, image_sizes=None) -> torch.FloatTensor:

        MAX_INPUT_ID = int(1e9)
        img_embeds = pixel_values
        img_sizes = image_sizes

        if self.img_features is not None:
            img_embeds = self.img_features.clone()
            self.img_features = None

        if self.img_sizes is not None:
            img_sizes = self.img_sizes

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        with torch.no_grad():
            positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=False)
        
        select = False

        if isinstance(self.img_projection, nn.Sequential):  
            target_device = self.img_projection[0].bias.device  
            target_dtype = self.img_projection[0].bias.dtype  
        else:  # It's a single nn.Linear layer  
            target_device = self.img_projection.bias.device  
            target_dtype = self.img_projection.bias.dtype  
        torch.set_printoptions(precision=10)
        if len(positions.tolist()) > 0:
            with torch.no_grad():
                g_values = abs(input_ids[positions[:, 0], positions[:, 1]])

            if self.use_hd_transform and img_sizes is not None and len(img_sizes):
                hd_transform = True
                assert img_embeds.ndim == 5, f'img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform'
                # img_embeds: (num_images, max_num_crops, 3, H, W)
                # img_sizes: (num_images, 2).view(1, -1)

                start_time = datetime.now()
                bs = img_embeds.shape[0]
                # Nx(HW)xC
                img_features = self.get_img_features(img_embeds.flatten(0, 1))
                base_feat_height = base_feat_width = int(img_features.shape[1] ** 0.5)

                assert base_feat_height == 24 and base_feat_width == 24, f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect 24x24 features for hd transform'

                # bs x max_num_crops x (24x24) x C
                img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
                print("imf mean",img_features.mean())
                C = self.image_dim_out
                H = base_feat_height

                output_imgs = []
                output_len = []
                # training is tensor, inference is list
                if isinstance(img_sizes, torch.Tensor):
                    img_sizes = img_sizes.view(-1, 2)
                for _bs in range(bs):
                    h, w = img_sizes[_bs]
                    h = h // 336 
                    w = w // 336
                    B_ = h * w

                    # 1 x (24x24) x 1024
                    global_img_feature = img_features[_bs, :1]

                    # 1 x 12 x 12 x 4096
                    glb_img = global_img_feature.reshape(1,H,H,C).reshape(1,H//2,2,H//2,2,C).contiguous().permute(0,1,3,2,4,5).reshape(1,H//2,H//2,4*C).contiguous()
                    print("glb init",glb_img.mean())
                    temp_glb_GN = self.sub_GN.repeat(1, H//2, 1, 1)
                    print("temp_glb_GN",temp_glb_GN.mean())

                    # 1 x 156 x 4096
                    glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1,-1,4*C)
                    print("glb cat",glb_img.mean())

                    # (max_num_crops-1) x (12x12) x C
                    sub_img = img_features[_bs, 1:]
                    print("sub_img",sub_img.mean())
                    
                    # 16x574x1024
                    # get rid of padding sub_img
                    sub_img = sub_img[:B_]
                    print("sub_img",sub_img.mean())

                    # (num_crops, 12, 2, 12, 2, 1024) -> (num_crops, 12, 12, 2, 2, 1024) -> (num_crops, 12*12, 4*1024)
                    sub_img = sub_img.reshape(B_,H,H,C).reshape(B_,H//2,2,H//2,2,C).contiguous().permute(0,1,3,2,4,5).reshape(B_,-1,4*C).contiguous()
                    sub_img = sub_img.reshape(1, h, w, 12, 12, -1).permute(0,1,3,2,4,5).reshape(1,h*12,w*12,4*C)
                    temp_sub_GN = self.sub_GN.repeat(1, h*12, 1, 1)
                    
                    print("temp_sub_GN",temp_sub_GN.mean())
                    
                    sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1,-1,4*C)
                    print("sub cat",sub_img.mean())
                    
                    # (1, num_img_tokens, 1024*4)

                    # glb + sub
                    if self.hd_transform_order == 'glb_sub':
                        output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                    elif self.hd_transform_order == 'sub_glb':
                        output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                    else:
                        raise NotImplementedError(f'hd_transform_order = {self.hd_transform_order}, not implemented')
                    print("last out", output_imgs[-1].mean())

                    temp_len = int((h*w+1)*144 + 1 + (h+1)*12)
                    assert temp_len == output_imgs[-1].shape[1], f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'
                    output_len.append(temp_len)
                
                num_img_tokens = output_len
                img_set_tensor = []
                for _output_img in output_imgs:
                    img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                    print("img_feature_proj",img_feature_proj.mean())
                    import numpy
                    numpy.save("layerout.npy",img_feature_proj.float().cpu().numpy())
                    print("^ LATEST")
                    
                    img_set_tensor.append(img_feature_proj)
                logger.info(f'img_embeds size: {img_embeds.size()}, image sizes: {img_sizes} loading time {datetime.now() - start_time}')
            elif img_embeds.ndim == 4:
                selected_g_values = g_values[::self.num_img_tokens]
                assert len(img_embeds) == len(selected_g_values), f'img_embeds size: {img_embeds.size()}, selected_g_values size: {len(selected_g_values)}, selected_g_value {selected_g_values}'
                start_time = datetime.now()
                tt = (
                    self.get_img_features(img_embeds)
                    .to(target_device)
                    .to(target_dtype)
                    .reshape(-1, self.image_dim_out)
                )
                logger.info(f'img_embeds size: {img_embeds.size()}, loading time {datetime.now() - start_time}')
                img_set_tensor = self.img_projection(tt)  # adapted visual features.
            elif img_embeds.ndim == 3:
                selected_g_values = g_values[::self.num_img_tokens]
                assert len(img_embeds) == len(selected_g_values), f'img_embeds size: {img_embeds.size()}, selected_g_values size: {len(selected_g_values)}, selected_g_value {selected_g_values}'
                tt = (
                    img_embeds
                    .to(target_device)
                    .to(target_dtype)
                    .view(-1, self.image_dim_out)
                )
                img_set_tensor = self.img_projection(tt)  # adapted visual features.
            else:
                raise NotImplementedError
            select = True
        
        with torch.no_grad():
            input_ids.clamp_min_(0).clamp_max_(self.vocab_size)
        
        hidden_states = self.wte(input_ids)

        if select:
            if hd_transform:
                idx = 0
                for i, cnt in enumerate(num_img_tokens):
                    print("hidden states 222", hidden_states.mean())
                    """if hidden_states.shape[1] != 1:
                        l = img_set_tensor[i].float().cpu().numpy()
                        import numpy
                        print("hidden_states mean",img_set_tensor[i].mean())
                        numpy.save("hidden_states_newest.npy", l)"""
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = (
                        img_set_tensor[i]
                        .to(hidden_states.dtype)
                        .to(hidden_states.device)
                        )
                    idx += cnt
            else:
                idx = 0
                assert len(selected_g_values) * self.num_img_tokens == len(img_set_tensor), f'len(selected_g_values) * self.num_img_tokens = {len(selected_g_values) * self.num_img_tokens}, len(img_set_tensor) = {len(img_set_tensor)}'
                for i, g in enumerate(selected_g_values):
                    cnt = self.num_img_tokens
                    hidden_states[positions[idx, 0], positions[idx, 1] : positions[idx, 1] + cnt] = (
                        img_set_tensor[i * cnt : (i + 1) * cnt]
                        .to(hidden_states.dtype)
                        .to(hidden_states.device)
                        )
                    idx += cnt
        
        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states
