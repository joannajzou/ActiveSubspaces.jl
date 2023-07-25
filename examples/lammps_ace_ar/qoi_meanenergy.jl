# define potential function
function Vz(Φ::Vector{Float64}, z::Vector{Float64}, ρ::MvNormal)
    θ = std_to_mvn(z, ρ)
    return Φ' * θ
end

function ∇xVz(ds::DataSet, z::Vector{Float64}, ρ::MvNormal)
    θ = std_to_mvn(z, ρ)
    ∇xΦ = reduce(vcat, get_values(get_force_descriptors(ds)))
    return vcat([dB' * θ for dB in ∇xΦ])
end

∇zVz(Φ::Vector{Float64}, z::Vector{Float64}, ρ::MvNormal) = Φ


# instantiate Gibbs object
V(x, z) = Vz(x, z, πβ)
∇xV(x, z) = ∇xVz(x, z, πβ)
∇θV(x, z) = ∇θVz(x, z, πβ)
πgibbs = Gibbs(V=V, ∇xV=∇xV, ∇θV=∇θV, β=1.0)

# define QoI as mean energy V
h(x, θ) = V(x, θ)
∇h(x, θ) = ∇θV(x, θ)


# define QoI
q = GibbsQoI(h=h, p=πgibbs)
