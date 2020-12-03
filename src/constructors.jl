function LTIQuadraticEnv(A,B,C,D,Q,R,
    s0::Union{Real,Vector{<:Real}},
    tspan::Tuple{<:Real,<:Real},
    dt::Real;
    #= Keyword arguments =#
    o_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for observation space
    o_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for observation space
    a_ub::Union{Nothing,Real,Vector{<:Real}}=nothing, # upper bound for action space
    a_lb::Union{Nothing,Real,Vector{<:Real}}=nothing, # lower bound for action space
    solver::DiffEqBase.AbstractODEAlgorithm=Tsit5(), 
    reltol::Real=1e-8, 
    abstol::Real=1e-8, # TODO: add more kwargs for integrator
    T=Float32
    )

    function ode(s, a, t) 
        ṡ = A * s + B * a
    end

    problem = ODEProblem(ode, s0, tspan)
    reward_fn = QuadraticReward(Q, R)
    observation_fn = LinearObservation(C, D)
    
    # Determine dimension of action
    if B isa Vector || B isa Real
        n_actions = 1
    elseif B isa Matrix
        n_actions = size(B)[2]
    else
        throw(ArgumentError("Input argument B is of type $(typeof(B))"))
    end 

    return DiffEqEnv(
        problem, 
        reward_fn,
        n_actions::Int,
        dt::Real;
        #= = Keyword arguments = =#
        observation_fn=observation_fn,
        o_ub=o_ub, o_lb=o_lb,
        a_ub=a_ub, a_lb=a_lb,
        solver=solver, reltol=reltol, abstol=abstol,
        T=T
    )
end