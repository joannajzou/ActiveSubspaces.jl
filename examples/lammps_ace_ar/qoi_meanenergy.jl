# define negative potential function
V(Φ::Vector{Float64}, θ::Vector{Float64}) = -Φ' * θ # with neg sign

function ∇xV(ds::DataSet, θ::Vector{Float64})
    ∇xΦ = reduce(vcat, get_values(get_force_descriptors(ds)))
    return vcat([-dB' * lp.β for dB in ∇xΦ])
end

∇θV(Φ::Vector{Float64}, θ::Vector{Float64}) = -Φ

# instantiate Gibbs object
πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)

# define QoI as mean energy V
h(x, θ) = V(x, θ)
∇h(x, θ) = ∇θV(x, θ)


# define QoI
q = GibbsQoI(h=h, p=πgibbs)
