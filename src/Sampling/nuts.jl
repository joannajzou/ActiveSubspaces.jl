"""
struct NUTS <: Sampler

Defines the struct with parameters of the No-U-Turn-Sampler algorithm for sampling.

# Arguments
- `ϵ :: Union{<:Real, Nothing}`        : step size
"""
struct NUTS <: Sampler
    ϵ :: Union{<:Real, Nothing}     # step size

    NUTS() = new(nothing)
    NUTS(ϵ::Float64) = new(ϵ)
    
end


# multi-d x: use AdvancedHMC
"""
function sample(lπ::Function, gradlπ::Function, sampler::NUTS, n::Int64, x0::Vector{<:Real})

Draw samples from target distribution π(x) using the No-U-Turn Sampler (NUTS) from the AdvancedHMC package. Assumes multi-dimensional support (x::Vector).

# Arguments
- `lπ :: Function`              : log likelihood of target π
- `gradlπ :: Function`          : gradient of log likelihood of π
- `sampler :: HMC`              : Sampler struct specifying sampling algorithm
- `n :: Int64`                  : number of samples
- `x0 :: Vector{<:Real}`        : initial state

# Outputs
- `samples::Vector{Vector}`     : vector of samples from π

"""
function sample(lπ::Function, gradlπ::Function, sampler::NUTS, n::Int64, x0::Vector{<:Real})
    D = length(x0)
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, lπ, gradlπ)

    if typeof(sampler.ϵ) == Nothing
        ϵ0 = find_good_stepsize(hamiltonian, x0)
        integrator = Leapfrog(ϵ0)
        proposal = HMCKernel(Trajectory{SliceTS}(integrator, ClassicNoUTurn())) 
        adaptor = NaiveHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        samples, _ = AdvancedHMC.sample(hamiltonian, proposal, x0, n, adaptor, nadapt)
    else
        integrator = Leapfrog(sampler.ϵ)
        proposal = HMCKernel(Trajectory{SliceTS}(integrator, ClassicNoUTurn()))
        samples, _ = AdvancedHMC.sample(hamiltonian, proposal, x0, n) # ; progress=false)
    end

    return samples
end


# 1-d x
"""
function sample(lπ::Function, gradlπ::Function, sampler::NUTS, n::Int64, x0::Real)

Draw samples from target distribution π(x) using the No-U-Turn Sampler (NUTS) for scalar-valued support (x::Real).

# Arguments
- `lπ :: Function`        : log likelihood of target π
- `gradlπ :: Function`    : gradient of log likelihood of π
- `sampler :: HMC`        : Sampler struct specifying sampling algorithm
- `n :: Int64`            : number of samples
- `x0 :: Real`            : initial state

# Outputs
- `samples :: Vector`     : vector of samples from π

"""
function sample(lπ::Function, gradlπ::Function, sampler::NUTS, n::Int64, x0::Real)
    x = Vector{Float64}(undef, n) # initialize state trajectory
    r = Vector{Float64}(undef, n) # initialize momentum trajectory
    x[1] = x0
    r[1] = randn()
    ϵ = sampler.ϵ

    # for iter = 1:10
    # xtemp = deepcopy(x)
    acceptances = 0

    # draw samples
    for i = 2:n
        r0 = randn()
        u = rand(Uniform(0, H(x[i-1], r0, lπ)))

        # initialize
        xm = x[i-1]; xp = x[i-1]; xi = x[i-1]
        rm = r0; rp = r0
        j = 0; n = 1; s = 1

        while s == 1
            # choose direction
            vj = rand([-1, 1]) 
            # make proposal
            if vj == -1 
                xm, rm, _, _, x̃, ñ, s̃ = buildtree(xm, rm, u, vj, j, ϵ, lπ, gradlπ)
            elseif vj == 1
                _, _, xp, rp, x̃, ñ, s̃ = buildtree(xp, rp, u, vj, j, ϵ, lπ, gradlπ)
            end
            # accept/reject proposal
            if s̃ == 1 
                flag = accept_or_reject( log(ñ) - log(n) )
                xi = flag ? x̃ : xi
                acceptances += flag
            end
            # update parameters
            x[i] = xi
            n += ñ
            s = s̃ * Ind((xp - xm)*rm >= 0) * Ind((xp - xm)*rp >= 0)
            j += 1
        end
    end
    accept_prob = acceptances / n

    return x
end

# build tree
function buildtree(x::Float64, r::Float64, u::Float64, vj::Int64, j::Int64, ϵ::Float64, lπ::Function, gradlπ::Function)
    Δmax = 1000 # fixed parameter
    L = 0

    if j == 0
    # base case: take one leapfrog step in direction of vj
        x̃, r̃ = leapfrog(x, r, vj*ϵ, gradlπ)
        ñ = Ind(u <= H(x̃,r̃,lπ))
        s̃ = Ind( log(H(x̃,r̃,lπ)) > (log(u) - Δmax) )
        L = 1
        return x̃, r̃, x̃, r̃, x̃, ñ, s̃, L
    else
        # recursion: implicitly build left and right subtrees 
        xm, rm, xp, rp, x̃, ñ, s̃ = buildtree(x, r, u, vj, j-1, ϵ, lπ, gradlπ)
        if s̃ == 1
            # make proposal
            if vj == -1 
                xm, rm, _, _, x̃2, ñ2, s̃2, _ = buildtree(xm, rm, u, vj, j-1, ϵ, lπ, gradlπ)
                L += 1
            elseif vj == 1
                _, _, xp, rp, x̃2, ñ2, s̃2, _ = buildtree(xp, rp, u, vj, j-1, ϵ, lπ, gradlπ)
                L += 1
            end
            # accept/reject proposal
            flag = accept_or_reject( log(ñ2) - log(ñ + ñ2) )
            x̃ = flag ? x̃2 : x̃
            # update parameters
            s̃ = s̃2 * Ind((xp - xm)*rm >= 0) * Ind((xp - xm)*rp >= 0)
            ñ += ñ2
        end
        return xm, rm, xp, rp, x̃, ñ, s̃, L
    end
end

# leapfrog simplectic integrator
function leapfrog(x::Float64, r::Float64, ϵ::Float64, gradlπ::Function)
    r̃ = r + (ϵ/2) * gradlπ(x)
    x̃ = x + ϵ * r̃
    r̃ = r̃ + (ϵ/2) * gradlπ(x̃)
    return x̃, r̃
end

# Hamiltonian
H(x::Float64, r::Float64, lπ::Function) = exp( lπ(x) .- 0.5 * r' * r )

