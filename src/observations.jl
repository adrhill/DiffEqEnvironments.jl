"""
Base struct for all DiffEqEnvironments.jl observations.
"""
struct ObservationFunction
    f::Function
end

(of::ObservationFunction)(s, a) = of.f(s, a)

"""
General observation of form ``o=of(s,a)``.
Passed function `of` requires a case `of(s, a::Nothing)`
    for ini
"""
CustomObservation(of) = ObservationFunction(of)

"""
General observation of form ``o=of(s)``.
"""
CustomStateObservation(of) = ObservationFunction((s, a) -> of(s))

"""
Full observation of state: ``o=s``.
"""
FullStateObservation() = ObservationFunction((s, a) -> s)

"""
Linear relationship of form ``o=\\mathbf{C}s`` between state and observation.
"""
LinearStateObservation(C) = ObservationFunction((s, a) -> C * s)

function LinearObservation(C, D)
    of(s, a::Nothing) = C * s # used for step 0
    of(s, a) = C * s + D * a
    return ObservationFunction(of)
end
