"""
Base struct for all DiffEqEnvironments.jl observations.
"""
struct ObservationFunction
    f
end

(of::ObservationFunction)(s) = of.f(s)

"""  
General observation of form ``o=f(s)``.
"""
function CustomObservationFunction(s)
    return ObservationFunction(s)
end

"""  
Full observation of state: ``o=s``.
"""
function FullObservationFunction()
    return ObservationFunction(identity) # returns its argument
end

"""  
Linear relationship of form ``o=\\mathbf{C}s`` between state and observation.
"""
function LinearObservationFunction(C)
    return ObservationFunction(s->C*s)
end