"""
Helper function to set action and observation spaces
"""
function _set_space(lb, ub, dim)
    # Check if types match
    typeof(lb) == typeof(ub) ||
        throw(ArgumentError("$(typeof(lb)) != $(typeof(ub)), types must match"))

    # Set defaults if no bounds are provided
    if isnothing(lb)
        if dim == 1
            ub = T(1e38)
            lb = -ub
        else
            ub = T(1e38) * ones(T, dim)
            lb = -ub
        end
    end

    # Check if sizes match
    length(lb) == length(ub) ||
        throw(ArgumentError("$(size(lb)) != $(size(ub)), size must match"))
    length(lb) == dim || throw(ArgumentError("$(size(lb)) != $(dim), size must match"))

    if lb isa Real
        return ClosedInterval{T}(lb, ub)
    elseif lb isa Vector{<:Real}
        return Space(ClosedInterval{T}.(lb, ub))
    end
end

"""
Helper function to set action and observation spaces
"""

function _set_space(lb::Real, ub::Real, dim, T)
    dim == 1 || throw(ArgumentError("$(size(lb)) != $(dim), size must match"))
    return ClosedInterval{T}(lb, ub)
end

function _set_space(lb::Vector{<:Real}, ub::Vector{<:Real}, dim, T)
    length(lb) == dim || throw(ArgumentError("$(size(lb)) != $(dim), size must match"))
    return Space(ClosedInterval{T}.(lb, ub))
end

function _set_space(lb::Nothing, ub::Nothing, dim, T)
    if dim == 1
        ub = T(1e38)
        lb = -ub
    else
        ub = T(1e38) * ones(T, dim)
        lb = -ub
    end
    return _set_space(lb, ub, dim, T)
end

function _set_space(lb::Nothing, ub::Union{Real,Vector{<:Real}}, dim, T)
    lb = oftype.(ub, -1e38)
    return _set_space(lb, ub, dim, T)
end

function _set_space(lb::Union{Real,Vector{<:Real}}, ub::Nothing, dim, T)
    ub = oftype.(lb, 1e38)
    return _set_space(lb, ub, dim, T)
end
