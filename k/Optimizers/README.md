# optimizer
- adjust learning rate
- improve learning efficient

## type of optimizers

### 1. Stochastic Gradient Descent
- random batch data
$$ W_{t+1} = W_t - \eta \nabla W_t $$

### 2. Momentum
- update step direction
$$ M_t = \gamma M_{t-1} + \eta \nabla W_t $$
$$ W_{t+1} = W_t - M_t $$

### 3. Adagrad
- update step size
$$ M_t = M_{t-1} + \nabla W_t ^2 $$
$$ W_{t+1} = \eta \frac{1}{\sqrt {M_t+\epsilon}} \nabla W_t $$

### 4. RMSprop
- Adagrad + regularization
$$ G_t = \gamma G_{t-1}+(1-\gamma)\nabla W_t^2 $$
$$ W_{t+1} = W_t - \frac{\eta}{\sqrt{G_t+\epsilon}}\nabla W_t $$

### 5. Adam
- RMSprop + Momentum + stabilization
$$ M_t = \beta_1 M_{t-1}+(1 - \beta_1) \nabla W_t $$
$$ V_t = \beta_2 V_{t-1}+(1 - \beta_2) \nabla W_t^2 $$
$$ W_{t+1} = W_t - \frac{\eta}{\sqrt{V_t+\epsilon}} \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} M_t $$
