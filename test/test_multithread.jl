using DiffEqEnvironments
using ReinforcementLearningBase
using ReinforcementLearningBase: test_interfaces!, test_runnable!
using ReinforcementLearningZoo
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
n_envs = 10
seed = 123

env = MultiThreadEnv([
    LTIQuadraticEnv(
        A, B, C, D, Q, R, s0, tspan, dt; a_lb=-1, a_ub=1, T=T, rng=StableRNG(hash(seed + i))
    ) for i in 1:n_envs
])

# Test whether necessary interfaces from RLBase are implemented correctly and consistently
test_interfaces!(env)
test_runnable!(env) #TODO: Currenty fails. Check if supposed to run on MultiThreadEnvs
