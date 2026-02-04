## DRC Extension: Temporal Residual Wrapper (for Execution Lag Compensation)
## No geometry, focus on pre-actuation via action history

class DRC_Temporal_Residual_Wrapper(nn.Module):
    def __init__(self, base_model, action_dim):
        """
        
        """
        super().__init__()
        self.base_model = base_model
        
        # 1. frozen Base Model 
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 2. input：origin obs + u_{t-1}
        base_input_dim = self.base_model.actor_mlp_t.mlp[0].in_features  # or actor_mlp
        residual_input_dim = base_input_dim + action_dim
        
        # 3. residual (velocity-level compensation)
        self.residual_mlp = nn.Sequential(
            nn.Linear(residual_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.base_model.mu_t.out_features)  #  output Δu (dim as mu_t)
        )
        
        # 4. NKP 
        nn.init.constant_(self.residual_mlp[-1].weight, 0.0)
        nn.init.constant_(self.residual_mlp[-1].bias, 0.0)

    def _get_residual_correction(self, obs_dict, prev_action):
        """ Δu_res"""
        obs = obs_dict['obs']  # 
        
        # MIX [obs, u_{t-1}]
        res_input = torch.cat([obs, prev_action], dim=-1)
        
        delta_u = self.residual_mlp(res_input)
        return delta_u

    @torch.no_grad()
    def act(self, obs_dict, prev_action):
        """sample_action (train/data collection)"""
        # Base mu/sigma/value
        base_mu, base_logstd, value = self.base_model._actor_critic(obs_dict)
        
        # （mu_t)
        delta_u = self._get_residual_correction(obs_dict, prev_action)
        
        # add (α scaling)
        new_mu = base_mu + delta_u  # or new_mu[:, :dim_t] += delta_u
        
        # sample
        sigma = torch.exp(base_logstd)
        distr = torch.distributions.Normal(new_mu, sigma)
        action = distr.sample()
        
        result = {
            'neglogpacs': -distr.log_prob(action).sum(1),
            'values': value,
            'actions': action,
            'mus': new_mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict, prev_action):
        """ (test/real robot)"""
        base_mu = self.base_model.act_inference(obs_dict)
        
        delta_u = self._get_residual_correction(obs_dict, prev_action)
        
        return base_mu + delta_u

    def forward(self, input_dict, prev_action):
        """train (loss ，residual)"""
        # Base output
        base_mu, base_logstd, value = self.base_model._actor_critic(input_dict)
        
        # residual
        delta_u = self._get_residual_correction(input_dict, prev_action)
        
        # mu
        new_mu = base_mu + delta_u
        
        # calculate
        sigma = torch.exp(base_logstd)
        distr = torch.distributions.Normal(new_mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(input_dict['prev_actions']).sum(1)
        
        return {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': new_mu,
            'sigmas': sigma,
        }