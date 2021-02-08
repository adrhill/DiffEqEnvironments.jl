using DiffEqEnvironments
using Documenter, Literate

## Use Literate.jl to generate docs and notebooks of examples
list_of_examples = ["example_lqr.jl"] # in /literate
for example in list_of_examples
    Literate.markdown( # markdown for Documenter.jl
        joinpath(@__DIR__, "literate", example),
        joinpath(@__DIR__, "src");
        documenter=true,
    )
    Literate.notebook( # markdown for Documenter.jl
        joinpath(@__DIR__, "literate", example),
        joinpath(@__DIR__, "notebooks"),
    )
end

## Build docs
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
    pages=["Home" => "index.md", "Examples" => "example_lqr.jmd"],
)

deploydocs(;
    repo="github.com/adrhill/DiffEqEnvironments.jl", devbranch="main", branch="gh-pages"
)
