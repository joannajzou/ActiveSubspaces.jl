include("gibbs.jl")


"""
function hasupdf(d::Distribution)

Returns 'true' if distribution d has an unnormalized pdf function (updf(), logupdf()).

# Arguments
- `d :: Distribution`    : distribution to check

# Outputs
- `flag :: Bool`        : true or false

"""
function hasupdf(d::Gibbs)
    return true
end

function hasupdf(d::Distribution)
    return false
end


export Gibbs, Gibbs!, params, updf, logupdf, normconst, gradlogpdf, hasupdf