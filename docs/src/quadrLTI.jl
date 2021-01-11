using DifferentialEquations
using LinearAlgebra
using ControlSystems

using Plots
include("plots.jl")

"""
Defines an SISO system of form
    ṡ = A*s + B*a
    o = C*s
This is also known as the rocket car problem, where
    s = [p, ṗ]  ,
p being the position of the car.

The action used as an input to the system is
the car's acceleration p̈ ∈ [-1, 1]. Therefore
    ṡ = [ṗ, p̈] = [0 1; 0 0] * s + [0; 1] * u .
Full observation of the state is given, as
    o = s   .
"""
T = Float32

# System matrices
A = [0 1; 0 0]
B = [0; 1]
C = [1 0; 0 1]
D = [0; 0]

# weights for quadratic cost function
Q = [1 0; 0 1]
R = [1][:, :]

# Set range of control input
a_lb = -1.0;
a_ub = 1.0;

# Define ODEProblem of SISO LTI system
function lti_ode(s, a, t)
    return ṡ = A * s + B * a
end

s0 = [-0.5f0, -0.5f0]
tspan = (0.0f0, 2.0f0)
prob_lti = ODEProblem(lti_ode, s0, tspan)

# Use ControlSystems.jl to solve continuous algebraic Riccati eq.
P = care(A, B, Q, R) # P encodes value function as v(s)= -s'*P*s/2
K = -R \ B' * P
println("K=", K)

v_lqr(s) = -s' * P * s / 2
π_lqr(s) = clamp(first(K * s), a_lb, a_ub) #  LQR controller w/ saturation

# Define closed-loop ODE
function cl_ode(π)
    return (s, _, t) -> lti_ode(s, π(s), t)
end

display(plot_full(π_lqr, v_lqr))

# Call constructor for LTI systems to try out LQR Policy
using DiffEqEnvironments
using ReinforcementLearningCore

dt = 0.1
env = LTIQuadraticEnv(A, B, C, D, Q, R, s0, tspan, dt; a_lb=-1, a_ub=1, T=T)
lqr_policy = LQRPolicy(A, B, Q, R, dt, a_lb, a_ub)
n_states, n_actions = 2, 1

hook = TotalRewardPerEpisode()
agent_lqr = Agent(;
    policy=lqr_policy,
    trajectory=CircularArraySARTTrajectory(;
        capacity=1000, state=Vector{T} => (n_states,), action=Vector{T} => (n_actions,)
    ),
)

run(agent_lqr, env, StopAfterEpisode(10), hook)
display(hook.rewards)

## DDPG example
# Create agent
using StableRNGs
using ReinforcementLearningBase
using ReinforcementLearningCore
using ReinforcementLearningZoo
using IntervalSets

seed = 123
rng = StableRNG(seed)
init = glorot_uniform(rng)
n_states = 2

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

stop_condition = StopAfterStep(10_000)

# Contruct composed hook
total_reward_per_episode = TotalRewardPerEpisode()
time_per_step = TimePerStep()

actor_losses = []
critic_losses = []
total_rewards = []

hook = ComposedHook(
    total_reward_per_episode,
    time_per_step,
    DoEveryNStep() do t, agent, env
        push!(actor_losses, agent.policy.actor_loss)
        push!(critic_losses, agent.policy.critic_loss)
        push!(total_rewards, total_reward_per_episode.reward)
    end,
)

run(agent, env, stop_condition, hook)

## Eval
display(plot([actor_losses, critic_losses, total_rewards]; layout=(3, 1)))

π_ddpg(s) = first(agent.policy.behavior_actor(s))
q_ddpg(s, a) = first(agent.policy.behavior_critic(vcat(s, a)))
v_ddpg(s) = q_ddpg(s, π_ddpg(s))

display(plot_full(π_ddpg, v_ddpg))
## Create environment with randomized initial conditions

s0_lb = [-1.0, -1.0]
s0_ub = [1.0, 1.0]
env_randomized = LTIQuadraticEnv(
    A, B, C, D, Q, R, s0_lb, s0_ub, tspan, dt; a_lb=-1, a_ub=1, T=T
)

actor_losses = []
critic_losses = []
total_rewards = []
stop_condition = StopAfterStep(10_000)

run(agent, env_randomized, stop_condition, hook)

display(plot_full(π_ddpg, v_ddpg))
