    pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add ActiveSubspaces to environment stack

using ActiveSubspaces
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(ActiveSubspaces, :DocTestSetup, :(using ActiveSubspaces); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "citations.bib"))

# Generate examples

#const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
#const OUTPUT_DIR = joinpath(@__DIR__, "src/generated")

#examples = Pair{String,String}[]

#for (_, name) in examples
#    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
#    Literate.markdown(example_filepath, OUTPUT_DIR, documenter = true)
#end

#examples = [title => joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(bib;
    modules = [ActiveSubspaces],
    authors = "joannajzou",
    repo = "https://github.com/cesmix-mit/ActiveSubspaces.jl/blob/{commit}{path}#{line}",
    sitename = "ActiveSubspaces.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://joannajzou.github.io/ActiveSubspaces.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    doctest = true,
    linkcheck = true,
    strict = false
)

deploydocs(;
    repo = "github.com/joannajzou/ActiveSubspaces.jl",
    devbranch = "main",
    push_preview = true
)
