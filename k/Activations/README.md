# activation

## why use?
- mappiing to higher dimmension for the linear seperablity. in other words, nonlinear mapping
- constraint output for the other calcuration


## type of activaitons

### 1. linear
$$ f(x) = x $$
$$ \nabla f(x) = 1 $$

### 2. sigmoid
$$ f(x) = \frac {1}{1+e^{-x}} $$
$$ \nabla f(x) = f(x) \cdot (1 - f(x)) $$

### 3. softmax
$$ f(x) = \frac {e^{x_i}} {\sum_{k=1}^N {e^{x_k}}} $$


### 4. relu
$$ f(x) = \begin{cases}
        0 & x < 0 \\
        x &  x \geq 0
        \end{cases} $$

$$ \nabla f(x) = \begin{cases}
        0 & x < 0 \\
        1 & x \geq 0
        \end{cases} $$

### 5. hyperbolic tangent
$$ f(x) = \frac {e^x - e^{-x}}{e^x + e^{-x}} $$
$$ \nabla f(x) = 1 - {f(x)}^2