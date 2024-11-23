#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch


class FeatureExtractorNetwork(nn.Module):
    def __init__(self, in_channels=5, out_dim=512+32):
        super(FeatureExtractorNetwork, self).__init__()
        # IMPORTANT:
        # # Save output dimensions, used to create the distributions
        # self.latent_dim_pi = out_dim
        # self.latent_dim_vf = out_dim
        self.frame_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 19, 512),
            nn.ReLU()
        )
        self.ob_net = nn.Sequential(
            nn.Linear(66, 32),
            nn.ReLU()
        )

    def assemble_obs(self, features):
        if features['P1_side'].ndim > 2:
            features['P1_side'], features['P2_side'] = features['P1_side'].squeeze(), features['P2_side'].squeeze()
            features['P1_stunned'], features['P2_stunned'] = features['P1_stunned'].squeeze(), features['P2_stunned'].squeeze()
        # print(features['P1_character'].shape, features['P2_character'].shape,
        #     features['P1_health'].shape, features['P2_health'].shape,
        #     features['P1_side'].shape, features['P2_side'].shape,
        #     features['P1_stun_bar'].shape, features['P2_stun_bar'].shape,
        #     features['P1_stunned'].shape, features['P2_stunned'].shape,
        #     features['P1_super_bar'].shape, features['P2_super_bar'].shape,
        #     features['P1_super_count'].shape, features['P2_super_count'].shape,
        #     features['P1_super_max_count'].shape, features['P2_super_max_count'].shape,
        #     features['P1_super_type'].shape, features['P2_super_type'].shape,
        #     features['stage'].shape, features['timer'].shape)
        # print(features['P1_character'], features['P2_character'],
        #     features['P1_health'], features['P2_health'],
        #     features['P1_side'], features['P2_side'],
        #     features['P1_stun_bar'], features['P2_stun_bar'],
        #     features['P1_stunned'], features['P2_stunned'],
        #     features['P1_super_bar'], features['P2_super_bar'],
        #     features['P1_super_count'], features['P2_super_count'],
        #     features['P1_super_max_count'], features['P2_super_max_count'],
        #     features['P1_super_type'], features['P2_super_type'],
        #     features['stage'], features['timer'])
        return torch.cat([
            features['P1_character'], features['P2_character'],
            features['P1_health'], features['P2_health'],
            features['P1_side'], features['P2_side'],
            features['P1_stun_bar'], features['P2_stun_bar'],
            features['P1_stunned'], features['P2_stunned'],
            features['P1_super_bar'], features['P2_super_bar'],
            features['P1_super_count'], features['P2_super_count'],
            features['P1_super_max_count'], features['P2_super_max_count'],
            features['P1_super_type'], features['P2_super_type'],
            features['stage'], features['timer']], dim=1)

    def forward(self, features):
        # print('1', features)
        frame = features['frame'].permute((0, 3, 1, 2))
        obs = self.assemble_obs(features)
        frame_feature = self.frame_net(frame)
        obs_feature = self.ob_net(obs)
        # print(frame_feature.shape, obs_feature.shape)
        combined_feature = torch.cat([frame_feature, obs_feature], dim=1)
        return combined_feature

    # def forward_actor(self, features):
    #     return self.forward(features)[0]
    #
    # def forward_critic(self, features):
    #     return self.value_net(features)[1]

class MLPNetwork(nn.Module):
    def __init__(self):
        super(MLPNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 256
        self.latent_dim_vf = 256
        self.policy_net = nn.Sequential(
            nn.Linear(544, 256),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(544, 256),
            nn.ReLU(),
        )
    def forward(self, features):
        # print('mlp', features.shape)
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        # print('mlp', features.shape)
        return self.policy_net(features)

    def forward_critic(self, features):
        # print('mlp', features.shape)
        return self.value_net(features)

class ActionNetwork(nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # self.latent_dim_pi = 90
        # self.latent_dim_vf = 90
        self.policy_net = nn.Sequential(
            nn.Linear(256, 90)
        )
    def forward(self, features):
        # print('2', features.shape)
        return self.policy_net(features)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # self.latent_dim_pi = 1
        # self.latent_dim_vf = 1
        self.value_net = nn.Sequential(
            nn.Linear(256, 1)
        )
    def forward(self, features):
        # print('3', features.shape)
        return self.value_net(features)
