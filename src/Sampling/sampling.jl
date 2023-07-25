abstract type Sampler end

include("mala.jl")
include("hmc.jl")
include("nuts.jl")


"""
function sample(lπ::Function, gradlπ::Function, sampler::Union{HMC,MALA}, n::Int64, x0::Real)

Draw samples from target distribution π(x) for scalar-valued support (x::Real).

# Arguments
- `lπ :: Function`                      : log likelihood of target π
- `gradlπ :: Function`                  : gradient of log likelihood of π
- `sampler :: Union{HMC,MALA}`          : Sampler struct specifying sampling algorithm
- `n :: Int64`                          : number of samples
- `x0 :: Real`                          : initial state

# Outputs
- `samples :: Vector`                   : vector of samples from π

"""
function sample(lπ::Function, gradlπ::Function, sampler::Union{HMC,MALA}, n::Int64, x0::Real)
    
    x = Vector{Float64}(undef, n)
    x[1] = x0
    ϵ = sampler.ϵ

    # for iter = 1:10
    # initialize
    xtemp = deepcopy(x)
    acceptances = 0

    # draw samples
    for i = 2:n
        xtemp[i], αi = propose(xtemp[i-1], lπ, gradlπ, sampler)
        acceptances += αi
    end
    acceptance_prob = acceptances / n
    println("ϵ = $(ϵ) \t α = $(acceptance_prob)")

    # # check acceptance prob
    # if 0.4 <= acceptance_prob <= 0.8
    #     x = deepcopy(xtemp)
    #     break
    # else
    #     if acceptance_prob < 0.6
    #         ϵ *= 0.9
    #     else
    #         ϵ *= 1.02
    #     end
    # end

    x = deepcopy(xtemp)

    return x

end


# Metropolis-Hastings step
function accept_or_reject(α :: Real)
    logα = min( 0.0, α )
    log(rand()) <= logα
end


# indicator function
Ind(α::Bool) = α ? 1 : 0



export MALA, HMC, NUTS, sample
