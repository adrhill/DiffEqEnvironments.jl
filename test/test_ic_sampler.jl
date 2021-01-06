using DiffEqEnvironments
using ReinforcementLearningBase
using IntervalSets
using Test

# Test uniform_sampler on multivariate ICs
s0_lb = -[1, 2, 3]
s0_ub = [4, 5, 6]
sampler = UniformSampler(s0_lb, s0_ub)

@test sampler isa DiffEqEnvironments.ICSampler
for _ in 1:10
    @test sampler() in Space(ClosedInterval{Float32}.(s0_lb, s0_ub))
end

# Test uniform_sampler on univariate ICs
s0_lb = -1
s0_ub = 2
sampler = UniformSampler(s0_lb, s0_ub)

@test sampler isa DiffEqEnvironments.ICSampler
for _ in 1:10
    @test sampler()[] in s0_lb..s0_ub
end

# Sample constant ICs by using lb = ub (used in default constructor)
s0_lb = [1, 2, 3]
s0_ub = s0_lb
sampler = UniformSampler(s0_lb, s0_ub)

@test sampler() == s0_ub

# Test whether ArgumentError triggers
s0_lb = [1, 2]
s0_ub = [1, 2, 3]
@test_throws ArgumentError UniformSampler(s0_lb, s0_ub)
