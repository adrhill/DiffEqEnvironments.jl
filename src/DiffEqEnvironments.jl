module DiffEqEnvironments

using ControlSystems: ss, lqr, state_space_validation, Continuous
using DiffEqBase: AbstractODEAlgorithm
using DifferentialEquations
using IntervalSets
using LinearAlgebra
using OrdinaryDiffEq: isadaptive
using Random
using ReinforcementLearningBase
using ReinforcementLearningCore

include("rewards.jl")
include("observations.jl")
include("ic_samplers.jl")
include("environment.jl")
include("constructors.jl")
include("policies.jl")

# Export reward functions
export SASReward, SAReward, ASReward, QuadraticReward, DecrementingReward

# Export observation functions
export CustomObservation, LinearObservation
export CustomStateObservation, LinearStateObservation, FullStateObservation

# Export IC samplers
export UniformSampler

# Export Policies
export FeedbackPolicy, LinearFeedbackPolicy, LQRPolicy

# Export environment
export DiffEqEnv
export LTIQuadraticEnv

end # module
