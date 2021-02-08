#=
# Linear quadratic problems

## The rocket car problem

To demonstrate the usefulness of LQR, we use the introductory example from [Daniel Liberzon's  *Calculus of Variations and Optimal Control Theory*](http://liberzon.csl.illinois.edu/publications.html#Books).
Adapted to RL notation, it reads:

> Consider a simple model of a car moving on a horizontal line.
Let $p \in R$ be the car’s position and let $a$ be the acceleration which acts as the control input. We put a bound on the maximal allowable acceleration by letting the action set $\mathcal{A}$ be the bounded interval $[−1, 1]$ (negative acceleration corresponds to braking). The dynamics of the car are $\ddot{p} = a$.
>
>In order to arrive at a first-order differential equation model, let us relabel the car’s position $p$ as $s_1$ and denote its velocity $\dot{p}$ by $s_2$ . This gives the control system $\dot{s}_1 = s_2$ , $\dot{s}_2 = a$ with state $s=\begin{pmatrix}s_1\\s_2\end{pmatrix} ∈ \mathbb{R}^2$.
>
>*Now, suppose that we want to “park” the car at the origin, i.e., bring it to rest there.*

The following animation shows a (very bad) attempt at solving the problem.
Let's do better using methods from Reinforcement Learning and Control Theory!

![](https://i.imgur.com/AzGqjL0.gif)

### Linear time-invariant quadratic problems

Writing down the first-order ODE, we are given
$$\dot{s}
	=\begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix} s
	+\begin{bmatrix}0\\1\end{bmatrix} a \quad .$$
This type of system is also called [*linear time-invariant* (LTI)](https://en.wikipedia.org/wiki/Linear_time-invariant_system), as the system state $s$ evolves as a constant, linear function of itself and actions $a$.
In this example, we use the quadratic expression
$$r(s, a) = -\left(s^T\mathbf{Q} s + a^T \mathbf{R} a\right)$$

as a dense reward function, as we want to park the car at $s=\underline{0}$ with acceleration $a=0$.[^QR-note]

[^QR-note]: By weighting the individual terms in $\mathbf{Q}$ and $\mathbf{R}$, we can enforce different behaviours. For example, high values of $\mathbf{R}$ will "punish" high accelerations $a$, leading to fuel savings.
After discretization of the system dynamics, this can be viewed as a [finite-horizon, discrete-time LQR problem](https://en.wikipedia.org/wiki/Linear–quadratic_regulator#Finite-horizon,_discrete-time_LQR).

## Creating the rocket car environment
### System dynamics
Writing down the linear system dynamics and weights for the reward function is straightforward:
=#

## System dynamics
A = [0 1; 0 0]
B = [0; 1]
C = [1 0; 0 1]
D = [0; 0]

## Quadratic reward
Q = [1 0; 0 1]
R = [1][:, :]

# We also define the range of control inputs as $\mathcal{A}=[-1,1]$ and set the range of initial conditions as $s_0\in[-1,1]^2$, from which we will uniformly sample.

## bounds on actions
a_lb = -1.0
a_ub = 1.0

## bounds on states
s0_lb = [-1.0, -1.0]
s0_ub = [1.0, 1.0]

# Finally, we need to specify the time horizon at the end of which the environment will be terminated and the discrete time step-size of the environment.

tspan = (0.0f0, 2.0f0)
dt = 0.1

# ### Creating an environment using `LTIQuadraticEnv`
# Instead of manually defining an ODE,
function lti_ode(s, a, t)
    return ṡ = A * s + B * a
end

# `DiffEqEnvironments` offers a simple constructor for LTI environments:

using DiffEqEnvironments

env = LTIQuadraticEnv(
    A, B, C, D, Q, R, s0_lb, s0_ub, tspan, dt; a_lb=a_lb, a_ub=a_ub, T=Float32
)

# ## Control theory: Linear quadratic regulator
# ### Algebraic Riccati equations with `ControlSystems.jl`
using DifferentialEquations
using LinearAlgebra
using ControlSystems

P = care(A, B, Q, R)
K = R \ B' * P
println("K=", K)

# Calculating the matrix $\mathbf{P}$ in a separate step is useful, as the state-value function is obtained for free as $v(s)= -s^T\mathbf{P}s$.

v_lqr(s) = -s' * P * s / 2

# The linear-quadratic regulator can now be applied as $\pi(s)=-\mathbf{K}s$. As a final step, we apply thresholds for input limits.

π_lqr(s) = clamp(first(-K * s), a_lb, a_ub)

# ### Using `LQRPolicy` as a baseline
# The same policy can also be obtained in `DiffEqEnv` as
lqr_policy = LQRPolicy(A, B, Q, R, dt, a_lb, a_ub)

# We can use `LQRPolicy` as part of an actor in a typical `ReinforcementLearning.jl` experiment:

using ReinforcementLearningCore

n_states, n_actions = 2, 1

hook = TotalRewardPerEpisode()
agent_lqr = Agent(;
    policy=lqr_policy,
    trajectory=CircularArraySARTTrajectory(;
        capacity=1000,
        state=Vector{Float32} => (n_states,),
        action=Vector{Float32} => (n_actions,),
    ),
)

run(agent_lqr, env, StopAfterEpisode(1), hook)

# ## Reinforcement Learning
#
# The following example is adapted from the `ReinforcementLearningZoo.jl` [example on DDPG on a pendulum](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl).
#
# ### DDPG agent
# After importing all necessary packages,

using StableRNGs
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningZoo
using IntervalSets

seed = 123
rng = StableRNG(seed)

# the neural networks for the actor-critic architecture are specified

init = glorot_uniform(rng)

function create_actor()
    return Chain(
        Dense(n_states, 30, relu; initW=init),
        Dense(30, 30, relu; initW=init),
        Dense(30, 1, tanh; initW=init),
    )
end

function create_critic()
    return Chain(
        Dense(n_states + 1, 30, relu; initW=init),
        Dense(30, 30, relu; initW=init),
        Dense(30, 1; initW=init),
    )
end

#  before calling RLZoo's `DDPGPolicy`:
agent = Agent(;
    policy=DDPGPolicy(;
        behavior_actor=NeuralNetworkApproximator(; model=create_actor(), optimizer=ADAM()),
        behavior_critic=NeuralNetworkApproximator(;
            model=create_critic(), optimizer=ADAM()
        ),
        target_actor=NeuralNetworkApproximator(; model=create_actor(), optimizer=ADAM()),
        target_critic=NeuralNetworkApproximator(; model=create_critic(), optimizer=ADAM()),
        γ=0.99f0,
        ρ=0.995f0,
        batch_size=64,
        start_steps=1000,
        start_policy=RandomPolicy(-1.0..1.0; rng=rng),
        update_after=1000,
        update_every=1,
        act_limit=1.0,
        act_noise=0.1,
        rng=rng,
    ),
    trajectory=CircularArraySARTTrajectory(;
        capacity=10000, state=Vector{Float32} => (n_states,), action=Float32 => ()
    ),
)

#=
### Running the experiment
`ReinforcementLearning.jl` evaluates experiments throught the use of "hooks". For the purpose of simplicity in these docs, a `ComposedHook` that pushes losses into empty arrays is used for logging.[^logging-note]

[^logging-note]: It is usually be a better choice to use `TensorBoardLogger.jl`, [as done in the DDPG pendulum example](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/335662aedc944146bba188f9f74edc4602c1e580/src/experiments/rl_envs/JuliaRL_DDPG_Pendulum.jl#L79-L97).
=#

## empty arrays for logging
actor_losses = []
critic_losses = []
total_rewards = []

## building-blocks for composed hook
total_reward_per_episode = TotalRewardPerEpisode()
time_per_step = TimePerStep()

hook = ComposedHook(
    total_reward_per_episode,
    time_per_step,
    DoEveryNStep() do t, agent, env
        push!(actor_losses, agent.policy.actor_loss)
        push!(critic_losses, agent.policy.critic_loss)
        push!(total_rewards, total_reward_per_episode.reward)
    end,
)

# We can now run the agent and visualize the
run(agent, env, StopAfterStep(10_000), hook)

## plot results
using Plots
plot([actor_losses, critic_losses, total_rewards]; layout=(3, 1))

# # Evaluating the agents
# To evaluate our policies, we visualize the value function as well es the phase-plot of out rocket-car environment in closed-loop with our policy.
#
# ## LQR
# Except for non-linearities introduced by clamping the output of the LQR-policy between ``[-1,1]``, the value function `v_lqr(s)` is an analytical solution to the linear time-invariant quadratic problem
include("plots.jl")
plot_full(π_lqr, v_lqr)

# ## DDPG
# When comparing our learned value function to this policy, we observe that
π_ddpg(s) = first(agent.policy.behavior_actor(s))
q_ddpg(s, a) = first(agent.policy.behavior_critic(vcat(s, a)))
v_ddpg(s) = q_ddpg(s, π_ddpg(s))

plot_full(π_ddpg, v_ddpg)
