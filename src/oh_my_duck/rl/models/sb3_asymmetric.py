"""Separate actor/critic inputs with native SB3 PPO and DictRolloutBuffer.

Only the actor's 61 sensor observations enter its network. Privileged simulator
observations are training-only and never enter the exported policy.
"""
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ObservationGroups(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        if set(observation_space.spaces) != {"actor", "critic"}:
            raise ValueError("Expected separate actor and critic observation groups")
        super().__init__(observation_space, sum(s.shape[0] for s in observation_space.spaces.values()))

    def forward(self, observations):
        return torch.cat((observations["actor"], observations["critic"]), dim=-1)


class SeparateMlp(nn.Module):
    def __init__(self, actor_dim, critic_dim, net_arch, activation_fn):
        super().__init__()
        self.actor_dim = actor_dim
        def network(size, widths):
            layers = []
            for width in widths:
                layers.extend((nn.Linear(size, width), activation_fn()))
                size = width
            return nn.Sequential(*layers), size
        self.policy_net, self.latent_dim_pi = network(actor_dim, net_arch["pi"])
        self.value_net, self.latent_dim_vf = network(critic_dim, net_arch["vf"])

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_dim])

    def forward_critic(self, features):
        return self.value_net(features[..., self.actor_dim:])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """Customize only SB3's networks; retain its distribution, loss and optimizer."""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", ObservationGroups)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = SeparateMlp(
            self.observation_space["actor"].shape[0],
            self.observation_space["critic"].shape[0],
            self.net_arch, self.activation_fn,
        ).to(self.device)

    def actor_mean(self, actor_observations):
        """Deterministic Gaussian action without constructing a privileged input."""
        return self.action_net(self.mlp_extractor.policy_net(actor_observations))
