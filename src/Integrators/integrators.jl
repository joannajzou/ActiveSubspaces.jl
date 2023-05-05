abstract type Integrator end

include("quadrature.jl")
include("monte_carlo.jl")
include("importance_sampling.jl")

export gausslegendre