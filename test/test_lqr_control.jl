using DiffEqEnvironments
using DifferentialEquations
using ReinforcementLearningBase
using ReinforcementLearningBase: test_interfaces!, test_runnable!
using ReinforcementLearningCore
using LinearAlgebra
using ControlSystems
using Test

"""
This test defines an SISO system of form
    ṡ = A*s + B*a
    o = C*s
for the rocket car problem, where
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

# weights for quadratic cost function
Q = [1 0; 0 1] * 10
R = [1][:,:]

# Set range of control input
a_lb = -1.0; a_ub = 1.0

# Set IC & tspan for integration
s0 = [-0.75, 0.0]
tspan = (0f0, 5f0)

# Define ODEProblem of SISO LTI system
function lti(s, a, t)
    ṡ = A * s + B * a
end
prob_lti = ODEProblem(lti, s0, tspan)

# Get LQR from ControlSystems.jl
K = lqr(A, B, Q, R)
π(s) = clamp.(-dot(K, s), a_lb, a_ub) #  LQR controller w/ saturation

# Define DiffEqEnv
r = QuadraticReward(Q, R)
n_actions = 1
dt = 0.1
env = DiffEqEnv(prob_lti, r, n_actions, dt; a_lb=a_lb, a_ub=a_ub, T=T)

# Manually step through environment
for _ ∈ 1:200
    s = RLBase.state(env)
    a = Float32.(π(s))[]
    env(a) # perform step
end

# Check if state [0,0] was reached
@test isapprox(env.state[1], 0, atol=1e-8)
@test isapprox(env.state[2], 0, atol=1e-8)

# Test whether necessary interfaces from RLBase are implemented correctly and consistently
reset!(env)
test_interfaces!(env)
test_runnable!(env)



# Run random policy using ReinforcementLearning.jl
lqr_policy = LQRPolicy(A, B, Q, R, dt, a_lb, a_ub)
n_states, n_actions = 2, 1

reset!(env)
hook = TotalRewardPerEpisode()
run(
    Agent(
        policy=lqr_policy,
        trajectory=CircularArraySARTTrajectory(
            capacity=1000,
            state=Vector{T} => (n_states,),
            action=Vector{T} => (n_actions,),
        ),
    ),
    env,
    StopAfterEpisode(10),
    hook
)
