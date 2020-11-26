module DiffEqEnvironments

using ReinforcementLearningBase
using DiffEqBase: AbstractODEAlgorithm
using DifferentialEquations

include("reward_function.jl")
include("observation_function.jl")
include("environment.jl")

# Export reward functions
export RewardFunction
export SASRewardFunction, SARewardFunction, ASRewardFunction
export QuadraticRewardFunction, DecrementingRewardFunction

# Export observation functions
export ObservationFunction, CustomObservationFunction
export FullObservationFunction, LinearObservationFunction

# Export environment
export DiffEqEnv

end # module
