import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    LayerNorm applied after each hidden Linear layer and before activation to prevent initialization issues
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Critic(nn.Module):
    '''
    Simple MLP Q-function.
        '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #self.net.apply(init_weights_xavier)

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # TODO: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)
        
        value = self.net(torch.cat([x, a], dim=-1))
        #####################################################################

        return value

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    '''
    Gaussian stochastic actor.
    Outputs a (mean, std) of a normal dist. 
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: define the network layers
        # The network should output TWO values for each action dimension:
        # The mean (mu) and the log standard deviation (log_std)
        # So, the final layer should have 2 * action_size outputs

        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2
        self.action_size = action_size

        self.net = MLP([obs_size]  + ([num_units] * num_layers)  + [2 * action_size])

            # Apply it
        self.net.apply(self._init_actor_weights)

    # Robust actor initialization – the only one that survives all seeds on this CartPole+MPO
    def _init_actor_weights(self, m):
        if isinstance(m, nn.Linear):
            # Orthogonal init with gain=1.0 for hidden layers
            # Final layer gets gain=0.01 for mean head, 1.0 for log_std head
            if m.out_features == 2 * self.action_size:
                # This is the final layer → split mean and log_std
                fan_in = m.weight.data.size(1)
                bound_mean = 0.01 / fan_in**0.5          # tiny weights for mean
                bound_std  = 1.0  / fan_in**0.5          # normal for log_std
                nn.init.uniform_(m.weight.data[:self.action_size], -bound_mean, bound_mean)   # mean head
                nn.init.uniform_(m.weight.data[self.action_size:], -bound_std,  bound_std)   # log_std head
                nn.init.zeros_(m.bias.data)  # zero bias = centered actions + log_std ≈ 0
            else:
                # Hidden layers: orthogonal + gain 1.0
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias.data)



    def forward(self, x):
        #####################################################################
        # TODO: code the forward pass
        # 1. Pass the observation x through the network
        # 2. Split the output into mean (mu) and log_std
        #    (Hint: use torch.chunk(..., 2, dim=-1))
        # 3. Constrain log_std to be within [LOG_STD_MIN, LOG_STD_MAX]
        #    (Hint: use torch.clamp)
        # 4. Calculate std = exp(log_std)
        # 5. Return a Normal distribution: torch.distributions.Normal(mu, std)

        mu, log_std = torch.chunk(self.net(x), 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        actions_gaussian = torch.distributions.Normal(mu, std)
        # CORRECT DEBUG PRINTS

        return actions_gaussian

class Agent:
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    torch.set_default_device(device)
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here
    tau: float = 0.005  # Polyak averaging coefficient
    learning_rate_q: float = 4e-4   # Learning rate for critics
    learning_rate_pi: float = 1.5e-4  # Learning rate for actor
    learning_rate_eta: float = 1e-3 # Learning rate for temperature eta
    target_kl: float = 0.15  # Target KL divergence for the M-step
    num_samples_q: int = 50  # Number of samples for E-step (critic update)
    num_samples_pi: int = 50 # Number of samples for M-step (actor update)
    num_layers_actor: int = 1 #1 good, 2 worse 3more worse
    num_units_actor: int = 76
    num_layers_critic: int = 2 #2 good, 1 worse, 3 to be tested
    num_units_critic: int = 128
    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor
    #########################################################################
    default = False
    if default:
        batch_size: int = 256
        gamma: float = 0.99  # MDP discount factor
        tau: float = 0.005  # Polyak averaging coefficient
        learning_rate_q: float = 3e-4  # Learning rate for critics
        learning_rate_pi: float = 1e-4  # Learning rate for actor
        learning_rate_eta: float = 1e-3 # Learning rate for temperature eta
        target_kl: float = 0.01  # Target KL divergence for the M-step
        num_samples_q: int = 20  # Number of samples for E-step (critic update)
        num_samples_pi: int = 20 # Number of samples for M-step (actor update)


    def __init__(self, env):
        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        #####################################################################
        # TODO: initialize actor, critic and attributes
        # 1. Initialize Actor (pi) and Target Actor (pi_target)
        # (Use the Actor class)
        self.pi = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers_actor, self.num_units_actor)
        self.pi_target = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers_actor, self.num_units_actor)
        # 2. Initialize TWO Critics (q1, q2) and their Targets (q1_target, q2_target)
        # (This is the "twin critics" trick from TD3, which MPO also uses)
        self.q1 = Critic(self.obs_size, self.action_size, self.num_layers_critic, self.num_units_critic)
        self.q2 = Critic(self.obs_size, self.action_size, self.num_layers_critic, self.num_units_critic)
        self.q1_target = Critic(self.obs_size, self.action_size, self.num_layers_critic, self.num_units_critic)
        self.q2_target = Critic(self.obs_size, self.action_size, self.num_layers_critic, self.num_units_critic)
        # 3. Initialize the optimizers for q1, q2, and pi
        
        self.q_optimizer = optim.Adam(list(self.q2.parameters()) + list(self.q1.parameters()), lr=self.learning_rate_q)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.learning_rate_pi)
        # 4. Initialize the learnable temperature 'eta' for the KL constraint
        self.log_eta = torch.tensor(1.0, device=self.device, requires_grad=True)
        self.eta_optimizer = optim.Adam([self.log_eta], lr=self.learning_rate_eta)

        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.step_counter = 0
        self.current_ep_return = -200
        self.episode_returns = []
        self.lr_reduced = False
        self.initial_lr_pi = self.learning_rate_pi
        self.safe_lr_pi = self.learning_rate_pi / 10


    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        # REWARD-DEPENDENT LR SCHEDULING — THIS IS THE HOLY GRAIL
        
        if not self.lr_reduced and (self.episode_returns[-1] if self.episode_returns else -200) >= -30:
            print(f"SUCCESS! Reducing actor LR {self.initial_lr_pi:.1e} → {self.safe_lr_pi:.1e}")
            for g in self.pi_optimizer.param_groups:
                g['lr'] = self.safe_lr_pi
            self.lr_reduced = True


        #####################################################################
        # TODO: code MPO training logic (E-Step and M-Step)

        with torch.no_grad():
            # 1. Get target policy distribution at next_obs
            # dist_target = self.pi_target(next_obs)
            dist_target = self.pi_target(next_obs)

            # 2. Sample K actions from the target policy distribution
            # (Hint: use .sample((self.num_samples_q,))
            next_actions_samples = dist_target.sample((self.num_samples_q,))
            # 3. Squash and rescale the K sampled actions
            next_actions_squashed = torch.tanh(next_actions_samples)
            next_actions_rescaled = next_actions_squashed * self.action_scale + self.action_bias
            
            # 4. Expand next_obs to match the K samples
            next_obs_expanded = next_obs.unsqueeze(0).repeat(self.num_samples_q, 1, 1)
            # 5. Get target Q-values for all (next_obs_expanded, next_actions_samples) pairs
            # (Use the twin target critics and take the minimum q_target_all = torch.min(self.q1_target(...), self.q2_target(...))
            q_target_all = torch.min(self.q1_target(next_obs_expanded, next_actions_rescaled), self.q2_target(next_obs_expanded, next_actions_rescaled))
            # 6. Average the K target Q-values to get the expected Q-value (q_next)
            # (Hint: reshape q_target_all to [num_samples, batch_size, 1] and take mean(dim=0))
            q_target_all = q_target_all.reshape(self.num_samples_q, self.batch_size, 1)
            q_next = q_target_all.mean(dim=0)
            # 7. Compute the final Bellman target (y)
            y = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * q_next


        # 8. Compute the critic loss (MSE loss) for BOTH critics
        q1_values = self.q1(obs, action)
        q2_values = self.q2(obs, action)

        q1_loss = nn.MSELoss()(q1_values, y)
        q2_loss = nn.MSELoss()(q2_values, y)
        q_loss = q1_loss + q2_loss
        
        # 9. Optimize the critic(s)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- Policy Improvement (M-Step): Update Actor & Eta ---
        # Goal: Find a new policy that maximizes Q-values,
        #       subject to a KL constraint.
        
        # 10. Get online policy distribution at obs
        dist_online = self.pi(obs)

        # 11. Sample K actions from the online policy distribution
        # (Use .rsample() for reparameterization)
        actions_gaussian = dist_online.rsample((self.num_samples_pi,))
        # 12. Squash and rescale the K sampled actions
        actions_tanh = torch.tanh(actions_gaussian)
        actions_squashed = actions_tanh * self.action_scale + self.action_bias
        # 13. Expand obs to match the K samples
        obs_expanded = obs.unsqueeze(0).repeat(self.num_samples_pi, 1, 1)
        # 14. Get Q-values for the (obs_expanded, actions_samples) pairs
        # (Use one of the online critics, e.g., q1. Detach             from critic graph)
        q1_values_samples = self.q1(obs_expanded, actions_squashed).detach()
        q2_values_samples = self.q2(obs_expanded, actions_squashed).detach()

        q_values_samples = torch.min(q1_values_samples, q2_values_samples)
        # 15. Get the temperature eta
        eta = torch.exp(self.log_eta).detach()
        # 16. Compute the importance weights (w)
        # (Hint: w = softmax(q_values_samples / eta, dim=0))
        w = torch.softmax(q_values_samples / eta, dim=0).squeeze(-1)

        # 17. Compute the log-probability of the sampled actions
        #     under the online policy
        # (This requires care with the Tanh squashing)
        # log_prob_samples = ...
        log_prob_gaussian = dist_online.log_prob(actions_gaussian).sum(-1)
        jacobian = torch.log(1 - actions_tanh.pow(2) + 1e-6).sum(-1)
        log_prob_samples = log_prob_gaussian - jacobian
        
        # 18. Compute the policy loss (weighted log-likelihood)
        # (Hint: pi_loss = - (w * log_prob_samples).mean())
        pi_loss = - (w * log_prob_samples).mean()
        # 19. Optimize the actor
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        
        # --- Temperature (eta) Update ---

        # 20. Get the new online distribution (post-update)
        dist_online_new = self.pi(obs)
        dist_online_new_det = torch.distributions.Normal(dist_online_new.loc.detach(), dist_online_new.scale.detach())
        
        #switched order of distributions???? #NOTE: it was working on one seed only maybe this is the issue
        kl = torch.distributions.kl.kl_divergence(dist_online_new_det, dist_online).mean()
        # 21. Compute the temperature loss
        # (Hint: eta_loss = torch.exp(self.log_eta) * (self.target_kl - kl).detach())
        eta_loss = torch.exp(self.log_eta) * (self.target_kl - kl).detach()
        
        # 22. Optimize eta
        self.eta_optimizer.zero_grad()
        eta_loss.backward()
        self.eta_optimizer.step()

        # --- Target Network Updates ---

        # 23. Softly update all target networks (q1_target, q2_target, pi_target)
        # (Use Polyak averaging with self.tau)
        def soft_update(target, source, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )
        soft_update(self.q1_target, self.q1, self.tau)
        soft_update(self.q2_target, self.q2, self.tau)
        soft_update(self.pi_target, self.pi, self.tau)
        ########################################################F#############

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # TODO: return the agent's action
        
        # 1. Convert obs to a tensor and send to device
        obs = torch.tensor(obs, device=self.device)
        # 2. Get the action distribution from the actor (self.pi)
        with torch.no_grad():
            # 3. Get action
            if train:
                # During training: sample from the distribution
                # (Hint: use dist.rsample() for reparameterization trick)
                action_gaussian = self.pi(obs).rsample()
            else:
                # During testing: use the deterministic mean
                action_gaussian = self.pi(obs).mean
            
            # 4. Squash the action through a Tanh function
            action_tanh = torch.tanh(action_gaussian)
            
            # 5. Rescale and shift the action to the environment's bounds
            # (Use self.action_scale and self.action_bias)
            action_scaled = action_tanh * self.action_scale + self.action_bias
            
            # 6. Convert to numpy array and return
            action = action_scaled.cpu().numpy().astype(np.float64)
        
        #####################################################################
        return action

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition

        self.current_ep_return += reward
        self.step_counter += 1
        # Detect episode end using the 200-step limit OR early termination
        if self.step_counter >= 200 or terminated:
            # Episode ended
            self.episode_returns.append(self.current_ep_return)
            print(f"Episode {len(self.episode_returns)} return: {self.current_ep_return:.1f}")
            # Reset for next episode
            self.current_ep_return = 0.0
            self.step_counter = 0

        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 30  # interactive episodes
    TEST_EPISODES = 30  # evaluation episodes
    save_video = False
    verbose = False
    seeds = np.arange(3)  # seeds for public evaluation
    print(f"seeds: {seeds}")

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)

        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        episode_training_data = {}

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)


        for i in range(TRAIN_EPISODES):
            episode_return = run_episode(env, agent, mode='train', verbose=verbose, rec=False)


        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
