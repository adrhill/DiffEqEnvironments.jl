"""
Base struct for all DiffEqEnvironments.jl observations.
"""
struct ObservationFunction
    f
end

(of::ObservationFunction)(s,a) = of.f(s, a)

"""  
General observation of form ``o=f(s,a)``.
"""
function CustomObservationFunction(of)
    return ObservationFunction(of)
end

"""  
Full observation of state: ``o=s``.
"""
function FullObservationFunction()
    return ObservationFunction((s, a) -> identity(s)) # return argument s
end

"""  
Linear relationship of form ``o=\\mathbf{C}s`` between state and observation.
"""
function LinearObservationFunction(C)
    return ObservationFunction((s, a) -> C * s)
end

function LinearObservationFunction(C, D)
    return ObservationFunction((s, a) -> C * s + D * a)
end