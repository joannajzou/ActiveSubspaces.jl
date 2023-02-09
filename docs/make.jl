using ActiveSubspaces
using Documenter

DocMeta.setdocmeta!(ActiveSubspaces, :DocTestSetup, :(using ActiveSubspaces); recursive=true)

makedocs(;
    modules=[ActiveSubspaces],
    authors="Joanna Zou <jjzou@mit.edu>",
    repo="https://github.com/joannajzou/ActiveSubspaces.jl/blob/{commit}{path}#{line}",
    sitename="ActiveSubspaces.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://joannajzou.github.io/ActiveSubspaces.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/joannajzou/ActiveSubspaces.jl",
    devbranch="main",
)
