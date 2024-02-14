## 01 - script for specifying the potential energy model and QoI

# potential energy function --------------------------------------------------------

# define ACE basis
nbody = 2
deg = 8
simdir = "ACE_MD_$(nbody)body_$(deg)deg_3/"

ace = ACE(species = [:Ar],         # species
          body_order = nbody,      # 2-body
          polynomial_degree = deg, # 8 degree polynomials
          wL = 1.0,                # Defaults, See ACE.jl documentation 
          csp = 1.0,               # Defaults, See ACE.jl documentation 
          r0 = 1.0,                # minimum distance between atoms
          rcutoff = 4.0)           # cutoff radius 


function V(Φ::Vector{Float64}, θ::Vector{Float64})
    return Φ' * θ
end

function ∇xV(ds::DataSet, θ::Vector{Float64})
    ∇xΦ = reduce(vcat, get_values(get_force_descriptors(ds)))
    return vcat([dB' * θ for dB in ∇xΦ])
end

∇θV(Φ::Vector{Float64}, θ::Vector{Float64}) = Φ


# instantiate Gibbs object --------------------------------------------------------
πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)


# parameter sampling density --------------------------------------------------------
μθ = JLD.load("fitted_params.jl")[β]
Σ0 = JLD.load("fitted_params.jl")[Σ]

d = length(μθ)
Σθ = 2e1*Σ0 + 1e-12*I(d)
ρθ = MvNormal(μθ, Σθ)

JLD.save("$(simdir)coeff_distribution.jld",
    "μ", μθ,
    "Σ", Σθ
)


# quantity of interest (QoI) -----------------------------------------------------
# mean energy V
h(x, θ) = V(x, θ)
∇h(x, θ) = ∇θV(x, θ)

# define QoI
q = GibbsQoI(h=h, ∇h=∇h, p=πgibbs)

