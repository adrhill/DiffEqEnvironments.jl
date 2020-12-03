module DiffEqEnvironments

using ControlSystems: state_space_validation
using ReinforcementLearningBase
using DiffEqBase: AbstractODEAlgorithm
using OrdinaryDiffEq: isadaptive
using DifferentialEquations

include("rewards.jl")
include("observations.jl")
include("environment.jl")
include("constructors.jl")
include("policies.jl")

# Export reward functions
export SASReward, SAReward, ASReward, QuadraticReward, DecrementingReward

# Export observation functions 
export CustomObservation, LinearObservation
export CustomStateObservation, LinearStateObservation, FullStateObservation

# Export Policies
export FeedbackPolicy, LinearFeedbackPolicy, LQRPolicy

# Export environment
export DiffEqEnv
export LTIQuadraticEnv

end # module