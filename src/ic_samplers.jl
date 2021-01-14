"""
Base struct for all DiffEqEnvironments.jl IC samplers.
"""
struct ICSampler
    sample::Function
end

(ics::ICSampler)() = ics.sample()

"""
Samples initial condition from Uniform distribution bounded by upper and lower bounds
"""
function UniformSampler(
    s0_lb::S, s0_ub::S; rng=Random.GLOBAL_RNG
) where {S<:Union{Real,Vector{<:Real}}}
    length(s0_lb) == length(s0_ub) || throw(
        ArgumentError(
            "Lengths $(length(s0_lb)) and $(length(s0_ub)) of bounds don't match."
        ),
    )
    return ICSampler(() -> rand(rng, length(s0_lb)) .* (s0_ub .- s0_lb) .+ s0_lb)
end
