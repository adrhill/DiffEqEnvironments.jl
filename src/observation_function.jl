"""
Base struct for all DiffEqEnvironments.jl observations.
"""
struct ObservationFunction
    f
end

(of::ObservationFunction)(s,a) = of.f(s, a)

"""  
General observation of form ``o=of(s,a)``.
Passed function `of` requires a case `of(s, a::Nothing)`
    for ini
"""
function CustomObservation(of)
    return ObservationFunction(of)
end

"""  
General observation of form ``o=of(s)``.
"""
function CustomStateObservation(of)
    return ObservationFunction((s, a) -> of(s))
end

"""  
Full observation of state: ``o=s``.
"""
function FullStateObservation()
    return ObservationFunction((s, a) -> s)
end

"""  
Linear relationship of form ``o=\\mathbf{C}s`` between state and observation.
"""
function LinearStateObservation(C)
    return ObservationFunction((s, a) -> C * s)
end

function LinearObservation(C, D)
    of(s, a::Nothing) = C * s # used for step 0
    of(s, a) = C * s + D * a
    return ObservationFunction(of)
end