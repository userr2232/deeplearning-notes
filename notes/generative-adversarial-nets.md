# Generative Adversarial Nets

## Related Work
* Undirected graphical models with latent variables
    * Restricted Boltzmann Machines
    * Deep Boltzmann Machines
* Deep Belief Networks
* Noise-Contrastive Estimation
* Generative Stochastic Network
* Variational Auto Encoders
* Stochastic Backpropagation

## Adversarial Nets
* Goal: Learn distribution $p_g$ over data $x$.
* Method:
    * Prior on input variables $p_z(z)$
    * Map $z \rightarrow G(z; \theta_g)$. G is differentiable.
    * Map $x \rightarrow D(x; \theta_d)$.
    * Train:
        * D to maximize the probability of classifying correctly its inputs (real or fake).
        * G to minimize $\log(1-D(G(z)))$.
        * In short: $\min\limits_{G}\max\limits_{D} V(G, D) = \mathbb{E}_{x\sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z\sim p_z(z)}[\log(1-D(G(z)))]$
        * It is preferable to train G by maximizing $\log D(G(z))$ to obtain stronger gradients at the beginning (by avoiding saturation).
* After several steps, the equilibrium (neither can make improvements) will be reached and $p_g = p_{data}$ (given that both networks have enough capacity).

## Theoretical Results
*This assumes models with infinite capacity in order to study convergence in the space of probability density functions.*

$$
\begin{equation}
\begin{align}
V(G, D) &= \mathbb{E}_{x\sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z\sim p_z(z)}[\log(1-D(G(z)))] \nonumber\\

&= \mathbb{E}_{x\sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{x\sim p_g(x)}[\log(1-D(x))] \nonumber\\

&= \int_{x}p_{data}(x)\log(D(x)) + p_g(x)\log(1-D(x))dx \nonumber
\end{align}
\end{equation}
$$


$(a, b)\in \mathbb{R}^2 \{0, 0\},\ y\rightarrow a\log(y) + b\log(1-y)$ achieves its maximum in $[0,1]$ at $\frac{a}{a+b}$ (derivative w.r.t. $y$ and make it $=0$). The discriminator does not need to be defined outside of $Supp(p_{data}) \cup Supp(p_{g})$ i.e. when $(a,b) = (0,0)$

$$C(G) = \max\limits_DV(G, D)$$