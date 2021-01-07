using DiffEqEnvironments
using ReinforcementLearningBase
using ReinforcementLearningBase: test_interfaces!, test_runnable!
using StableRNGs
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
s0 = [0.0, 0.0]
tspan = (0.0, 5.0)
dt = 0.1
T = Float32
env = LTIQuadraticEnv(
    A, B, C, D, Q, R, s0, tspan, dt; a_lb=-1, a_ub=1, T=T, rng=StableRNG(123)
)

@test state(env) == C * s0 # test initial observation

# Test whether necessary interfaces from RLBase are implemented correctly and consistently
test_interfaces!(env)
test_runnable!(env)
