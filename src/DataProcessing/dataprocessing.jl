"""
function mvn_to_std(v::Vector, ρ::MvNormal)

Converts a general multivariate normal variable to standard multivariate normal N(0,I).

# Arguments
- `v :: Vector`    : sample from multivariate normal distribution ρ
- `ρ :: MvNormal`  : multivariate normal distribution N(μ,Σ)

# Outputs
- `z :: Vector`    : sample transformed to be from standard normal distribution N(0,I)

"""
function mvn_to_std(v::Vector, ρ::MvNormal)
    L = cholesky(ρ.Σ).L
    z = pinv(L) * (v - ρ.μ) 
    return z
end

"""
function std_to_mvn(v::Vector, ρ::MvNormal)

Converts a standard multivariate normal variable to general multivariate normal N(μ,Σ).

# Arguments
- `z :: Vector`    : sample from standard normal distribution N(0,I)
- `ρ :: MvNormal`  : multivariate normal distribution N(μ,Σ)

# Outputs
- `v :: Vector`    : sample transformed to be from ρ

"""
function std_to_mvn(z::Vector, ρ::MvNormal)
    L = cholesky(ρ.Σ).L
    v = ρ.μ + L*z
    return v
end


"""
function nat_to_unf(z::Vector, ρ::Uniform)

Converts a general uniform variable from U[a,b] to a natural multivariate variable from U[-1,1].

# Arguments
- `v :: Vector`    : sample whose elements are from ρ
- `ρ :: Uniform`  : general uniform distribution U[a,b]

# Outputs
- `v :: Vector`    : sample transformed to be from natural distribution U[-1,1]

"""
function unf_to_nat(v::Vector, ρ::Uniform)
    D = (ρ.b - ρ.a)*I(length(z))
    return inv(D)*(2*z - (ρ.a + ρ.b))
end


"""
function nat_to_unf(z::Vector, ρ::Uniform)

Converts a natural multivariate variable from U[-1,1] to a general uniform variable from U[a,b].

# Arguments
- `z :: Vector`    : sample whose elements are from U[-1,1]
- `ρ :: Uniform`  : general uniform distribution U[a,b]

# Outputs
- `v :: Vector`    : sample transformed to be from ρ

"""
function nat_to_unf(z::Vector, ρ::Uniform)
    D = (ρ.b - ρ.a)*I(length(z))
    return 0.5 * (D*z + (ρ.a + ρ.b))
end



export mvn_to_std, std_to_mvn, unf_to_nat, nat_to_unf