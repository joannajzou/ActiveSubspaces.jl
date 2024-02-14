include("gibbs.jl")
include("mixture.jl")

"""
function hasupdf(d::Distribution)

Returns 'true' if distribution d has an unnormalized pdf function (updf(), logupdf()).

# Arguments
- `d :: Distribution`    : distribution to check

# Outputs
- `flag :: Bool`        : true or false

"""
hasupdf(d::Gibbs) = true

hasupdf(d::Distribution) = false

"""
function hasapproxnormconst(d::Distribution)

Returns 'true' if distribution d requires an approximation of its normalizing constant.
# Arguments
- `d :: Distribution`    : distribution to check

# Outputs
- `flag :: Bool`        : true or false

"""
hasapproxnormconst(d::Gibbs) = true

hasapproxnormconst(d::MixtureModel{Union{Univariate,Multivariate}, Continuous, Gibbs}) = true

hasapproxnormconst(d::Uniform) = false

# pdf(d::Distribution, x, integrator::Nothing) = pdf(d, x)


export Gibbs, Gibbs!, params, updf, logupdf, normconst, gradlogpdf, hasupdf, hasapproxnormconst