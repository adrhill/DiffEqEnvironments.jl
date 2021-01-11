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
Q = [1 0; 0 1] * 10
R = [1][:, :]

# Set range of control input
a_lb = -1.0;
a_ub = 1.0;

# Define ODEProblem of SISO LTI system
function lti_ode(s, a, t)
    return ṡ = A * s + B * a
end

s0 = [-0.5f0, -0.5f0]
tspan = (0.0f0, 5.0f0)
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

p1 = plot_phase_portrait(cl_ode(π_lqr))
p2 = plot_trajectory(cl_ode(π_lqr), s0)
p3 = plot_value(v_lqr)

display(plot(p1, p2, p3; layout=(1, 3), size=(900, 250)))

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
