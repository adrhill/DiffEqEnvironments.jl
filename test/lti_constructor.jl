using DiffEqEnvironments
using LinearAlgebra
using ReinforcementLearning
using Test

# LTI system matrices
A = I * 0.1
B = [1; 1]
C = [1 1]
D = 0

# Weights for quadratic cost function
Q = I * 10
R = I

# Call constructor for LTI systems
s0 = [0f0, 0f0]
tspan = (0f0, 5f0)
dt = 0.1
env = LTIQuadraticEnv(A,B,C,D,Q,R,s0,tspan,dt)

# Run random policy using ReinforcementLearning.jl
hook = TotalRewardPerEpisode()
run(
    Agent(
        ;policy = RandomPolicy(env),
        trajectory = VectCompactSARTSATrajectory(
            state_type=Bool,
            action_type=Any,
            reward_type=Int,
            terminal_type=Bool,
        ),
    ),
    env,
    StopAfterEpisode(10),
    hook
)