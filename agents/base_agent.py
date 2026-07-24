"""
agents/base_agent.py
Minimal PPO implementation shared by physio_agent, behavior_agent,
context_agent. Built on raw torch only (no stable-baselines3/gymnasium
dependency — avoided due to environment install restrictions).

Each agent treats risk classification as an RL problem:
  state  = feature vector for one window/sample
  action = predicted class (0 = non-stress/non-depressed, 1 = stress/depressed)
  reward = +1 if action matches true label, -1 otherwise
  confidence = softmax probability of the chosen action
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # logits


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOAgent:
    def __init__(self, input_dim: int, n_actions: int = 2, lr: float = 3e-4,
                 gamma: float = 0.99, clip_eps: float = 0.2, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(input_dim, n_actions=n_actions).to(self.device)
        self.value = ValueNetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )
        self.gamma = gamma
        self.clip_eps = clip_eps

    def select_action(self, state: np.ndarray):
        """Returns action, log_prob, confidence (softmax prob of chosen action)."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(state_t)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        confidence = probs[0, action.item()].item()
        return action.item(), log_prob.item(), confidence

    def predict(self, state: np.ndarray):
        """Greedy inference (no sampling) — used at eval/aggregation time."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_t)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action].item()
        return action, confidence

    def predict_proba_positive(self, state: np.ndarray) -> float:
        """Return P(action=1) specifically — needed for correct AUC computation."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_t)
            probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item()

    def update(self, states, actions, old_log_probs, rewards, epochs: int = 4):
        """Single-step episodic PPO update (each sample = one-step episode)."""
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # normalize rewards as advantage proxy (single-step episodes → return == reward)
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        for _ in range(epochs):
            logits = self.policy(states_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.value(states_t)
            value_loss = nn.functional.mse_loss(values, rewards_t)

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(), "value": self.value.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])