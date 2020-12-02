module DiffEqEnvironments

using ReinforcementLearningBase
using DiffEqBase: AbstractODEAlgorithm
using OrdinaryDiffEq: isadaptive
using DifferentialEquations

include("reward_function.jl")
include("observation_function.jl")
include("environment.jl")
include("lti_quadratic.jl")

# Export reward functions
export SASReward, SAReward, ASReward, QuadraticReward, DecrementingReward

# Export observation functions 
export CustomObservation, LinearObservation
export CustomStateObservation, LinearStateObservation, FullStateObservation

# Export environment
export DiffEqEnv
export LTIQuadraticEnv

end # module