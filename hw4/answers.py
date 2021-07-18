r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp["batch_size"] = 64
    hp["learn_rate"] = 5e-3
    hp["eps"] = 2e-8
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp["beta"] = 0.2
    hp["learn_rate"] = 8e-3
    hp["num_workers"]=2
    # ========================
    return hp


part1_q1 = r"""
The baseline is simply the mean of the predicted q values, as an empicical expectation of the q value, that help to reduce variance.
If we look at the reward as a function $R(t)$ this function is not continues with time (there is a huge diffrance between sparse reawrds and dense rewards but it's not the scope of the answer).
The point is as such: when the agent play a trajectory with a high reward, the gradients will be very small as oppose to other trajectories.
when we substract the expected q values, we would reduce this effect (the variance between trajectories).
The idea of a baseline can also help us determine a higher baseline and try to help the model update the parameters even if in general the reward is positive (as in the launar lander example).
"""


part1_q2 = r"""

$ v_{\pi}(s) = \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \cdot \tilde{q}_{\pi}(s,a) $.

Recall that under policy $\pi$:

$$
\begin{align}
v_{\pi}(s) &= \mathbb{E}[g_t(\tau))|s_t=s, \pi] \\
q_{\pi}(s,a) &= \mathbb{E}[g_t(\tau)|s_t=s, a_t = a, \pi]
\end{align}
$$  

the relation between $v_{\pi}(s)$ and $q_{\pi}(s,a)$ can be presented as:  
$$
\begin{align}
v_{\pi}(s) &= \mathbb{E}[g_t(\tau)|s_t=s, \pi] \\ &= \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \mathbb{E}[g_t(\tau)|s_t=s, a_t = a, \pi] 
\\ &= \sum_{a \in \mathcal{A}(s)} \pi(a|s_t=s) \cdot q_{\pi}(s,a)
\end{align}
$$

and the estimation relay on that relation.
"""


part1_q3 = r"""

1. First experiment: Result analysis
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_p</span> shows the policy-gradient as the negative average loss across trajectories.  
      Both $\hat\grad\mathcal{L}_{\text{PG}}(\vec{\theta})$ `vpg` and $\hat\grad\mathcal{L}_{\mathcal{H}(\pi)}(\vec{\theta})$ `epg` start from a high loss and gradually improve their policy (shown as loss values steadily climbing towards $0$).  
      Gradient derivations which subtract the baseline: $\hat\grad\mathcal{L}_{\text{BPG}}(\vec{\theta})$ `bpg` and $\hat\grad\mathcal{L}_{\text{CPG}}(\vec{\theta})$ `cpg` exhibit minimal fluctuations around $0$.  
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_e</span> shows the entropy loss as the negative entropy. The values rise as the networks learns with subsequent iterations. High entropy values imply that the action probability distribution resembles a uniform distribution, whereas a reduction in entropy marks the network's converges on the effective policy; the network is more confident of the proper actions.  
    The subtraction of baseline $b$ in $\hat\grad\mathcal{L}_{\text{CPG}}(\vec{\theta})$ `cpg` speeds up the rate of convergence (in comparison to $\hat\grad\mathcal{L}_{\mathcal{H}(\pi)}(\vec{\theta})$ `epg`) thus making the training process more efficient.
    * The graph <span style="font-family:Courier; font-size:1.2em;">baseline</span> illustrates that the baseline $b$ (the mean of batch q-values) increases with every batch. 
    $$\hat{q}_{t} = \sum_{t'\geq t} \gamma^{t'-t}r_{t'+1}.$$
    It is crucial for maintaining low variance that the magnitude of the baseline $b$ accounts for the increasing rewards.
    * The graph <span style="font-family:Courier; font-size:1.2em;">mean\_reward</span> shows an increase in mean reward, indicating that all evaluated losses were effective (to various degrees) for learning to solve the task. The introduction of entropy loss had little impact on the outcome of training, possibly due to the small action space. This conclusion stems from the similarity in mean reward between `epg` and `vpg` (no baseline, with and without entropy loss) and between `cpg` and `bpg` (baseline, with and without entropy loss). Also note that employing a baseline had a significant impact on the efficacy of the training process - manifesting in better mean rewards for `bpg` and `cpg`. 


2. Comparison between Regular Policy-Gradient `cpg` and Advantage Actor-Critic `AAC`
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_p</span> shows that `AAC` obtains a lower trajectory loss compared to `cpg` .  
        This is attributed to `AAC` being a more expressive model that better captures a reliable approximation of state values.
    * The graph <span style="font-family:Courier; font-size:1.2em;">loss\_e</span> illustrates the entropy loss of `AAC` and `cpg` .  
    The two methods perform similarly - up to a scale factor determined by the ratio of entropy loss multipliers $~{}^{\mathbb{\beta}_{\text{CPG}}~~}{\mskip -5mu/\mskip -3mu}_{~~\mathbb{\beta}_{\text{AAC}}}$.
    * The graph <span style="font-family:Courier; font-size:1.2em;">mean\_reward</span> shows similar performance for `AAC` and `cpg` .  
    This can be attributed to the ability of the simple `cpg` (Policy-gradient with baseline) to reliably comprehend the task:  
    The task is simple enough as to not mandate the more expressive `AAC` approach.


"""
