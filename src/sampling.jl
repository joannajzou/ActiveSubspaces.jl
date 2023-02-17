abstract type Sampler end


struct HMC <: Sampler
    L :: Int                        # length of integrating Hamiltonian's equations
    ϵ :: Float64                    # step size
end


# multi-d x: use AdvancedHMC
function sample(lπ::Function, gradlπ::Function, sampler::HMC, n::Int64, x0::Vector{<:Real})
    D = length(x0)
    metric = DiagEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, lπ, gradlπ)

    integrator = Leapfrog(sampler.ϵ)
    proposal = HMCKernel(StaticTrajectory(integrator, sampler.L))
    samples, _ = AdvancedHMC.sample(hamiltonian, proposal, x0, n; progress=false)

    return samples
end


struct NUTS <: Sampler
    ϵ :: Union{<:Real, Nothing}     # step size

    NUTS() = new(nothing)
    NUTS(ϵ::Float64) = new(ϵ)
    
end


# multi-d x: use AdvancedHMC
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
        samples, _ = AdvancedHMC.sample(hamiltonian, proposal, [x0], n; progress=false)
    end

    return samples
end


# 1-d x
function sample(V::Function, ∇V::Function, sampler::NUTS, n::Int64, x0::Real)
    burn = Int(0.3 * n) # number of burn-in steps
    x = Vector{Float64}(undef, n + burn) # initialize state trajectory
    r = Vector{Float64}(undef, n + burn) # initialize momentum trajectory
    x[1] = x0
    r[1] = randn()
    ϵ = sampler.ϵ

    # for iter = 1:10
    # xtemp = deepcopy(x)
    acceptances = 0

    # draw samples
    for i = 2:n + burn
        r0 = randn()
        u = rand(Uniform(0, H(x[i-1], r0, V)))

        # initialize
        xm = x[i-1]; xp = x[i-1]; xi = x[i-1]
        rm = r0; rp = r0
        j = 0; n = 1; s = 1

        while s == 1
            # choose direction
            vj = rand([-1, 1]) 
            # make proposal
            if vj == -1 
                xm, rm, _, _, x̃, ñ, s̃ = buildtree(xm, rm, u, vj, j, ϵ, V, ∇V)
            elseif vj == 1
                _, _, xp, rp, x̃, ñ, s̃ = buildtree(xp, rp, u, vj, j, ϵ, V, ∇V)
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

    return x[(burn+1):end]
end

# build tree
function buildtree(x::Float64, r::Float64, u::Float64, vj::Int64, j::Int64, ϵ::Float64, V::Function, ∇V::Function)
    Δmax = 1000 # fixed parameter
    L = 0

    if j == 0
    # base case: take one leapfrog step in direction of vj
        x̃, r̃ = leapfrog(x, r, vj*ϵ, ∇V)
        ñ = Ind(u <= H(x̃,r̃,V))
        s̃ = Ind( log(H(x̃,r̃,V)) > (log(u) - Δmax) )
        L = 1
        return x̃, r̃, x̃, r̃, x̃, ñ, s̃, L
    else
        # recursion: implicitly build left and right subtrees 
        xm, rm, xp, rp, x̃, ñ, s̃ = buildtree(x, r, u, vj, j-1, ϵ, V, ∇V)
        if s̃ == 1
            # make proposal
            if vj == -1 
                xm, rm, _, _, x̃2, ñ2, s̃2, _ = buildtree(xm, rm, u, vj, j-1, ϵ, V, ∇V)
                L += 1
            elseif vj == 1
                _, _, xp, rp, x̃2, ñ2, s̃2, _ = buildtree(xp, rp, u, vj, j-1, ϵ, V, ∇V)
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
function leapfrog(x::Float64, r::Float64, ϵ::Float64, ∇V::Function)
    r̃ = r + (ϵ/2) * ∇V(x)
    x̃ = x + ϵ * r̃
    r̃ = r̃ + (ϵ/2) * ∇V(x̃)
    return x̃, r̃
end

# Hamiltonian
H(x::Float64, r::Float64, V::Function) = exp( V(x) .- 0.5 * r' * r )

# Metropolis-Hastings step
function accept_or_reject(α :: Real)
    logα = min( 0.0, α )
    log(rand()) <= logα
end

# indicator function
Ind(α::Bool) = α ? 1 : 0