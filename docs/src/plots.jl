"""
Plot phase portrait of state s=(p, ṗ)
"""
function plot_phase_portrait!(plt, ode; n=15, scale=0.075, c=:black)
    ṡ(s) = ode(s, 0, 0) * scale
    ṡ(p, ṗ) = ṡ([p, ṗ])

    ps = range(-1, 1; length=n)
    ṗs = range(-1, 1; length=n)

    pps = [p for p in ps for ṗ in ṗs]
    ṗṗs = [ṗ for p in ps for ṗ in ṗs]
    return quiver!(
        plt,
        pps,
        ṗṗs;
        quiver=ṡ,
        arrow=(style = :closed; headlength = 1),
        lw=1,
        c=c,
        title="Phase portrait",
        xlabel="s",
        ylabel="ṡ",
        xlim=(-1, 1),
        ylim=(-1, 1),
        show=true,
    )
end

plot_phase_portrait(ode; kwargs...) = plot_phase_portrait!(plot(), ode; kwargs...)

"""
Plot continuous state-value-function v(s)
"""
function plot_value!(plot, v; levels=10)
    ps = range(-1, 1; length=100)
    ṗs = range(-1, 1; length=100)
    vp(p, ṗ) = v([p, ṗ])

    return contour(
        ps,
        ṗs,
        vp;
        levels=levels,
        fill=true,
        title="Value function",
        xlabel="s",
        ylabel="ṡ",
        xlim=(-1, 1),
        ylim=(-1, 1),
        show=true,
    )
end

plot_value(v; kwargs...) = plot_value!(plot(), v; kwargs...)

"""
Plot actions selected by deterministic policy π
"""
function plot_actions!(plot, π; levels=10)
    ps = range(-1, 1; length=100)
    ṗs = range(-1, 1; length=100)
    πp(p, ṗ) = π([p, ṗ])

    return contour(
        ps,
        ṗs,
        πp;
        levels=levels,
        color=:vik,
        fill=true,
        title="Action a=π(s)",
        xlabel="s",
        ylabel="ṡ",
        xlim=(-1, 1),
        ylim=(-1, 1),
        show=true,
    )
end

plot_actions(π; kwargs...) = plot_actions!(plot(), π; kwargs...)

"""
Plot trajectory in state space
"""
function plot_trajectory!(plt, ode, s0; tspan=(0.0f0, 5.0f0), saveat=0.01, c=:black, lw=3)
    prob = ODEProblem(ode, s0, tspan)
    sol = solve(prob, Tsit5(); saveat=saveat)

    return plot!(
        plt,
        sol[1, :],
        sol[2, :];
        lw=lw,
        c=c,
        line=:arrow,
        title="Trajectory",
        xlabel="s",
        ylabel="ṡ",
        xlim=(-1, 1),
        ylim=(-1, 1),
        legend=false,
        show=true,
    )
end

plot_trajectory(ode, s0; kwargs...) = plot_trajectory!(plot(), ode, s0; kwargs...)
