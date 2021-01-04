"""
Plot phase portrait of state s=(p, ṗ)
"""
function plot_phase_portrait!(plt, ode; n=15, scale=0.075, c=:black)
    ṡ(s) = ode(s, 0, 0) * scale
    ṡ(p, ṗ) = ṡ([p,ṗ])

    ps = range(-1, 1, length=n)
    ṗs = range(-1, 1, length=n)

    pps = [p for p in ps for ṗ in ṗs]
    ṗṗs = [ṗ for p in ps for ṗ in ṗs]
    quiver!(plt, pps, ṗṗs, quiver=ṡ,
        arrow=(style = :closed; headlength = 1),
        xlim=(-1, 1), ylim=(-1, 1),
        xlabel="s", ylabel="ṡ",
        lw=1, c=c, show=true)
end

plot_phase_portrait(ode; kwargs...) = plot_phase_portrait!(plot(), ode; kwargs...)

"""
Plot continuous state-value-function v(s)
"""
function plot_value!(plot, v; levels=10)
    ps = range(-1, 1, length=100)
    ṗs = range(-1, 1, length=100)
    vp(p, ṗ) = v([p,ṗ]) 

    contour(ps, ṗs, vp, levels=levels, fill=true, 
        xlabel="s", ylabel="ṡ", show=true)
end

plot_value(v; kwargs...) = plot_value!(plot(), v; kwargs...)

"""
Plot trajectory in state space
"""
function plot_trajectory!(plt, ode, s0; tspan=(0f0, 5f0), saveat=0.01, c=:black, lw=3)
    prob = ODEProblem(ode, s0, tspan)
    sol = solve(prob, Tsit5(), saveat=saveat)

    plot!(plt, sol[1,:], sol[2,:], lw=lw, c=c, line=:arrow,
        xlim=(-1, 1), ylim=(-1, 1), 
        xlabel="s", ylabel="ṡ",
        legend=false, show=true)
end

plot_trajectory(ode, s0; kwargs...) = plot_trajectory!(plot(), ode, s0; kwargs...)