using DiffEqEnvironments
using Documenter

makedocs(;
    modules=[DiffEqEnvironments],
    authors="Adrian Hill",
    repo="https://github.com/adrhill/DiffEqEnvironments.jl/blob/{commit}{path}#L{line}",
    sitename="DiffEqEnvironments.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://adrhill.github.io/DiffEqEnvironments.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/adrhill/DiffEqEnvironments.jl",
)
