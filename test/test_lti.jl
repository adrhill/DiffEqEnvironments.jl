using DiffEqEnvironments
using LinearAlgebra
using ReinforcementLearning
using Test

# LTI system matrices: n_state=2, n_observations=2, n_actions=1
A = [1 0; 0 1] * 0.1
B = [0.1; 0.1] 
C = [1 0; 0 1]
D = [0; 0]

# Weights for quadratic cost function
Q = [1 0; 0 1] * 10
R = [1]

# Call constructor for LTI systems
s0 = [0f0, 0f0]
tspan = (0f0, 5f0)
dt = 0.1
env = LTIQuadraticEnv(A, B, C, D, Q, R, s0, tspan, dt; a_lb=-1, a_ub=1)

@test get_state(env) == C * s0 # test initial observation


# Run random policy using ReinforcementLearning.jl
hook = TotalRewardPerEpisode()
run(
    Agent(
        ;policy = RandomPolicy(env),
        trajectory = VectCompactSARTSATrajectory(
            state_type=Any,
            action_type=Any,
            reward_type=Real,
            terminal_type=Bool,
        ),
    ),
    env,
    StopAfterEpisode(10),
    hook
)