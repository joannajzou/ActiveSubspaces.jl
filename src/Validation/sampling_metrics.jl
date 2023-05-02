using StatsBase

"""
function EffSampleSize(hsamp::Vector)

Computes the effective sample size of a set of nMC samples from an MCMC chain. 

# Arguments
- `hsamp :: Vector`   : nMC-vector of d-dimensional function evaluation h

# Outputs
- `ESS :: Float64`    : effective sample size (where 1 < ESS < nMC)
"""
function EffSampleSize(hsamp::Vector)
    
    hsamp = reduce(hcat, hsamp)' # nsamp-by-dim matrix
    n = size(hsamp, 1)
    θ = (1 .+ 2*sum(autocor(hsamp), dims=1))'
    ESS = Int(floor(minimum(n ./ θ)))
    return ESS
end


"""
function ISWeightVariance(wsamp::Vector)

Returns the variance of importance sampling weights. 

# Arguments
- `wsamp :: Vector`   : nMC-vector of weights

# Outputs
- `var(wsamp) :: Float64`    : variance of weights
"""
function ISWeightVariance(wsamp::Vector)
    return var(wsamp)
end


"""
function ISWeightVariance(wsamp::Vector)

Returns the effective sample size from importance sampling weights. 

# Arguments
- `wsamp :: Vector`   : nMC-vector of weights, either 1-dim or d-dim. (for modified weights)

# Outputs
- `ESS :: Float64`    : minimum ESS across dimensions of wsamp
"""

function ISWeightESS(wsamp::Vector)
    d = length(wsamp[1])

    if d == 1
        return sum(wsamp)^2 / sum( wsamp.^2 ) 
    else
        wsamp = reduce(hcat, wsamp)'
        return minimum([ sum(wsamp[:,i]).^2 / sum(wsamp[:,i].^2) for i = 1:d ]) # modified ESS 
    end
end
    