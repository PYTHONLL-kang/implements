# 1. Gradient Descent
$$ W := W - \alpha \cdot \frac{\partial L}{\partial W} $$

# 2. Back Propagation
$$f(x) = w1x + b1
g(x) = w2x + b2
y = f(g(x))
\frac{\partial y}{\partial g(x)} = \frac{\partial y}{\partial f(x)} \cdot w1
\frac{\partial y}{\partial w1} = \frac{\partial y}{\partial f(x)} \cdot g(x)
...$$
