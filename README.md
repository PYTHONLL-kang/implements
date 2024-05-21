# 1. Gradient Descent
$$ W := W - \alpha \cdot \frac{\partial L}{\partial W} $$

# 2. Back Propagation
### 1. set two layer dnn
$f(w1, x) = w1 \cdot x$

$\sigma (x) = \frac{1}{1+e^{-x}}$

$g(w2, x) = w2 \cdot x$

$\hat y = f(w1, \sigma (g(w2, x))) = w1 \cdot \sigma (g(w2, x))$

### 2. set object function

$J(\hat y) = (y-\hat y)^2$

### 3. update w1

- $w1 := w1 - \alpha \cdot \frac{\partial J}{\partial w1}$

$\frac{\partial J}{\partial w1} = \frac{\partial J}{\partial \hat y} \cdot \frac{\partial \hat y}{\partial w1}$

$\frac{\partial J}{\partial \hat y} = 2(y-\hat y)$

$\frac{\partial \hat y}{\partial w1} = \sigma (g(w,x))$

### 4. update w2

- $w2 := w2 - \alpha \cdot \frac{\partial J}{\partial w2}$

$\frac{\partial J}{\partial w2} = \frac{\partial J}{\partial \hat y} \cdot \frac{\partial \hat y}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial w2}$

$\frac{\partial \hat y}{\partial \sigma} = w1 \cdot \frac{\partial \sigma}{\partial g}$

$\frac{\partial \sigma}{\partial g} = \sigma (g(w2,x)) \cdot (1-\sigma (g(w2, x)))$

$\frac{\partial \sigma}{\partial g} = x$