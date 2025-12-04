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


class Critic(nn.Module):
    '''
    Simple MLP Q-function.
        '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # DONE: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)
        value = self.net(torch.cat([x, a], dim=-1))
        return value

LOG_STD_MAX = 2 #default 2
LOG_STD_MIN = -5 #default -5

class Actor(nn.Module):
    '''
    Gaussian stochastic actor.
    Outputs a (mean, std) of a normal dist. 
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # DONE: define the network layers
        # The network should output TWO values for each action dimension:
        # The mean (mu) and the log standard deviation (log_std)
        # So, the final layer should have 2 * action_size outputs

        self.net = MLP([obs_size] + ([num_units] * num_layers) + [2 * action_size])

        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # DONE: code the forward pass
        # 1. Pass the observation x through the network
        # 2. Split the output into mean (mu) and log_std
        mu, log_std = torch.chunk(self.net(x), 2, dim=-1)
        #    (Hint: use torch.chunk(..., 2, dim=-1))
        # 3. Constrain log_std to be within [LOG_STD_MIN, LOG_STD_MAX]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #    (Hint: use torch.clamp)
        # 4. Calculate std = exp(log_std)
        std = torch.exp(log_std)
        # 5. Return a Normal distribution: torch.distributions.Normal(mu, std)
        return torch.distributions.Normal(mu, std)


class Agent:
    # INSERT_YOUR_CODE
    # automatically select compute device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # set device to default for torch if not otherwise specified for all torch objects
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)

    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    batch_size: int = 256 #default 256
    gamma: float = 0.99  # MDP discount factor

    tau              : float = 0.01  # Polyak averaging coefficient
    learning_rate_q  : float = 3e-4   # Learning rate for critics
    learning_rate_pi : float = 1e-4   # Learning rate for actor
    learning_rate_eta: float = 1e-3   # Learning rate for temperature eta
    target_kl        : float = 0.01   # Target KL divergence for the M-step
    num_samples_q    : int   = 30     # Number of samples for E-step (critic update) #default 20
    num_samples_pi   : int   = 30     # Number of samples for M-step (actor update) #default 20
    num_layers_actor : int   = 2
    num_units_actor  : int   = 64
    num_layers_critic: int   = 2
    num_units_critic : int   = 64
    
    #########################################################################

    #check non default values for hyperparameters
    target_kl        : float = 0.1   # Target KL divergence for the M-step
    

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        print(f"action_low:\n {self.action_low}")
        print(f"action_high:\n {self.action_high}")
        print(f"action_scale:\n {self.action_scale}")
        print(f"action_bias:\n {self.action_bias}")


        #####################################################################
        # DONE: initialize actor, critic and attributes
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
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.learning_rate_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.learning_rate_q)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.learning_rate_pi)
        # 4. Initialize the learnable temperature 'eta' for the KL constraint
        self.log_eta = torch.tensor(1.0, device=self.device, requires_grad=True)
        self.eta_optimizer = optim.Adam([self.log_eta], lr=self.learning_rate_eta)

        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
        
        #debug variables
        self.debug_gradients = {}

    def _debug_gradients_NN(self, NN, name= None,verbose=False):
        with torch.no_grad():
            grad_sum = 0
            for param in NN.parameters():
                grad_sum += param.grad.norm().item() if param.grad is not None else 0
            if verbose:
                print(f"grad_sum: {grad_sum}")
            if name is not None:
                if name not in self.debug_gradients:
                    self.debug_gradients[name] = []
                self.debug_gradients[name].append(grad_sum)
    
    def _debug_gradients_value(self, value, name= None,verbose=False):
        with torch.no_grad():
            if verbose:
                print(f"value: {value}")
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            if name is not None:
                if name not in self.debug_gradients:
                    self.debug_gradients[name] = []
                self.debug_gradients[name].append(value)
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        # Ensure reward and done are [batch_size, 1]
        reward = reward.view(self.batch_size, 1)
        done = done.view(self.batch_size, 1)

        self.train_step += 1

        #####################################################################
        # DONE: code MPO training logic (E-Step and M-Step)

        with torch.no_grad():
            # 1. Get target policy distribution at next_obs
            dist_target = self.pi_target(next_obs)
            self._debug_gradients_value(dist_target.mean, name="dist_target")
            # 2. Sample K actions from the target policy distribution
            # (Hint: use .sample((self.num_samples_q,))
            next_actions_samples = dist_target.sample((self.num_samples_q,))
            # 3. Squash and rescale the K sampled actions
            next_actions_samples = torch.tanh(next_actions_samples) * self.action_scale + self.action_bias
            # 4. Expand next_obs to match the K samples
            next_obs_expanded = next_obs.unsqueeze(0).repeat(self.num_samples_q, 1, 1)
            # 5. Get target Q-values for all (next_obs_expanded, next_actions_samples) pairs
            # (Use the twin target critics and take the minimum q_target_all = torch.min(self.q1_target(...), self.q2_target(...))
            q_target_all = torch.min(self.q1_target(next_obs_expanded, next_actions_samples), self.q2_target(next_obs_expanded, next_actions_samples))
            self._debug_gradients_value(q_target_all, name="q_target_all")
            # 6. Average the K target Q-values to get the expected Q-value (q_next)
            # (Hint: reshape q_target_all to [num_samples, batch_size, 1] and take mean(dim=0))
            q_target_all = q_target_all.reshape(self.num_samples_q, self.batch_size, 1)
            q_next = q_target_all.mean(dim=0)
            # 7. Compute the final Bellman target (y)
            y = reward + (1 - done) * self.gamma * q_next
            self._debug_gradients_value(y, name="y")
        # 8. Compute the critic loss (MSE loss) for BOTH critics
        q1_values = self.q1(obs, action)
        q2_values = self.q2(obs, action)
        q1_loss = nn.MSELoss()(q1_values, y)
        q2_loss = nn.MSELoss()(q2_values, y)
        
        # 9. Optimize the critic(s)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        #print(f"q1_loss: {q1_loss.item()}")
        self._debug_gradients_NN(self.q1, name="q1")
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        #print(f"q2_loss: {q2_loss.item()}")
        self._debug_gradients_NN(self.q2, name="q2")
        self.q2_optimizer.step()

        # --- Policy Improvement (M-Step): Update Actor & Eta ---
        # Goal: Find a new policy that maximizes Q-values,
        #       subject to a KL constraint.
        
        # 10. Get online policy distribution at obs
        #NOTE: used pi_target instead of pi (lagging, not online)
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
        # (Use one of the online critics, e.g., q1. Detach from critic graph)
        q1_values_samples = self.q1(obs_expanded, actions_squashed).detach()
        q2_values_samples = self.q2(obs_expanded, actions_squashed).detach()
        q_values_samples = torch.min(q1_values_samples, q2_values_samples)
        self._debug_gradients_value(q_values_samples, name="q_values_samples")
        # 15. Get the temperature eta
        eta = torch.exp(self.log_eta).detach()
        # 16. Compute the importance weights (w)
        # (Hint: w = softmax(q_values_samples / eta, dim=0))
        w = torch.softmax(q_values_samples / eta, dim=0).squeeze(-1)
        # 17. Compute the log-probability of the sampled actions
        #     under the online policy
        # (This requires care with the Tanh squashing)
        # log_prob_samples = ...
        # Reparameterize the SAME pre-tanh values under the online policy
        log_prob_gaussian = dist_online.log_prob(actions_gaussian).sum(-1)   # sum dims
        # Tanh correction
        jacobian = torch.log(1 - actions_tanh.pow(2) + 1e-6).sum(-1)
        log_prob_samples = log_prob_gaussian - jacobian
        self._debug_gradients_value(log_prob_samples, name="log_prob_samples")

        # 18. Compute the policy loss (weighted log-likelihood)
        # (Hint: pi_loss = - (w * log_prob_samples).mean())
        pi_loss = - (w * log_prob_samples).mean()
        self._debug_gradients_value(pi_loss, name="pi_loss")
        # 19. Optimize the actor
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self._debug_gradients_NN(self.pi, name="pi")
        self.pi_optimizer.step()
        # --- Temperature (eta) Update ---

        # 20. Get the new online distribution (post-update)
        dist_online_new = self.pi(obs)
        self.kl = torch.distributions.kl_divergence(dist_online, dist_online_new).mean()
        self._debug_gradients_value(self.kl, name="kl")
        # 21. Compute the temperature loss
        # (Hint: eta_loss = torch.exp(self.log_eta) * (self.target_kl - kl).detach())
        grad_kl = (self.target_kl - self.kl).detach()
        eta_loss = torch.exp(self.log_eta) * grad_kl
        self._debug_gradients_value(grad_kl, name="grad_kl")
        self._debug_gradients_value(eta_loss, name="eta_loss")
        # 22. Optimize eta
        self.eta_optimizer.zero_grad()
        eta_loss.backward()
        #print(f"eta_loss: {eta_loss.item()}")
        #print("Eta grad:", self.log_eta.grad.item() if self.log_eta.grad is not None else "None")
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
        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # DONE: return the agent's action
        
        # 1. Convert obs to a tensor and send to device
        obs = torch.tensor(obs, device=self.device).float()
        # 2. Get the action distribution from the actor (self.pi)
        dist = self.pi(obs)
        with torch.no_grad():
            # 3. Get action
            if train:
                # During training: sample from the distribution
                # (Hint: use dist.rsample() for reparameterization trick)
                action_gaussian = dist.rsample()
            else:
                # During testing: use the deterministic mean
                action_gaussian = dist.mean
            
            # 4. Squash the action through a Tanh function
            action_tanh = torch.tanh(action_gaussian)
            
            # 5. Rescale and shift the action to the environment's bounds
            # (Use self.action_scale and self.action_bias)
            action_scaled = action_tanh * self.action_scale + self.action_bias
            # 6. Convert to numpy array and return
            action = action_scaled.cpu().numpy().astype(np.float64)
        return action


    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


DEBUG_MODE = False


# ==================== DEBUG MAIN (put at the very bottom of solution.py) ====================
if __name__ == "__main__" and DEBUG_MODE:
    import torch
    import numpy as np
    from utils import ReplayBuffer

    torch.manual_seed(0)
    np.random.seed(0)

    # --------------------- Toy MDP (1D, fully known) ---------------------
    # State  : s ∈ ℝ (we'll use s = [0.5] or s = [-0.3])
    # Action : a ∈ [-2, 2] (squashed from tanh)
    # Dynamics: s' = s + a + noise(0, 0.1)
    # Reward  : r = -|s'|   (optimal policy pushes s → 0)
    # True optimal Q*(s,a) = -|s + a| - γ * something small ≈ -|s + a|
    # True optimal policy: μ*(s) = -s  (push exactly opposite to current state)

    obs_size = 1
    action_size = 1
    action_low = torch.tensor([-2.0])
    action_high = torch.tensor([2.0])

    # --------------------- Create agent with your current code ---------------------
    class DummyEnv:
        observation_space = type('obj', (), {'shape': (obs_size,)})
        action_space = type('obj', (), {'shape': (action_size,), 'low': np.array([-2.0]), 'high': np.array([2.0])})

    env = DummyEnv()
    agent = Agent(env)                     # ← your Agent class (with all your bugs/fixes)
    agent.buffer = ReplayBuffer(100_000, obs_size, action_size, agent.device)

    # --------------------- Fill replay buffer with decent data ---------------------
    # We'll generate transitions from a slightly exploratory policy: μ = -0.8*s, σ = 0.5
    for _ in range(5000):
        s = np.random.randn(1) * 1.0
        a_gauss = -0.8 * s + np.random.randn(1) * 0.5
        a = np.tanh(a_gauss) * 2.0                      # squash to [-2, 2]
        s_next = s + a + np.random.randn(1) * 0.1
        r = -abs(s_next.item())
        done = False

        agent.store((s.astype(np.float32),
                     a.astype(np.float32),
                     np.array([r], dtype=np.float32),
                     s_next.astype(np.float32),
                     np.array([done], dtype=np.float32)))

    print("Replay buffer filled with 5000 decent transitions")
    print(f"Initial eta = {torch.exp(agent.log_eta).item():.4f}")
    print(f"Target KL = {agent.target_kl}")

    # --------------------- Run 200 training steps and watch everything ---------------------
    for step in range(200):
        agent.train()          # ← this is the only thing we call

        if step % 20 == 0 or step < 10:
            # Sample a known state
            with torch.no_grad():
                test_s = torch.tensor([[0.5]], device=agent.device, dtype=torch.float32)
                dist = agent.pi(test_s)
                mu, std = dist.mean.item(), dist.scale.item()
                a_raw = dist.mean
                a_tanh = torch.tanh(a_raw)
                a_final = a_tanh * agent.action_scale + agent.action_bias

                # True optimal action for s=0.5 → a = -0.5 (push left)
                optimal_a = -0.5

                print(f"\nStep {step:3d} | "
                      f"eta={torch.exp(agent.log_eta).item():6.3f} | "
                      f"KL={agent.kl.item():6.3f} | "
                      f"μ={mu:6.3f} std={std:6.3f} → a={a_final.item():6.3f} "
                      f"(optimal ≈ {optimal_a})")
                for key, value in agent.debug_gradients.items():
                    print(f"   Mean {key}: {np.mean(value):.3f}")
                    print(f"   Std {key}: {np.std(value):.3f}")




# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__' and not DEBUG_MODE:

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(1)  # seeds for public evaluation

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

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)
        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

            for key, value in agent.debug_gradients.items():
                print(f"   Mean {key}: {np.mean(value):.3f}")
                print(f"   Std {key}: {np.std(value):.3f}")



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