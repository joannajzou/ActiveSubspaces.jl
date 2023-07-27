
"""
function MCSE(q::GibbsQoI, xsamp::Vector)

Estimates the Monte Carlo standard error (se = std(h(x))/√n) using an empirical estimate of standard deviation.

# Arguments
- `q :: GibbsQoI`     : Gibbs QoI struct
- `xsamp :: Vector`   : vector of samples

# Outputs
- `se :: Float64`     : Monte Carlo standard error
"""
function MCSE(q::GibbsQoI, xsamp::Vector)
    hsamp = q.h(xsamp)
    n = length(xsamp)
    σ2 = 1/(n-1) * sum(hsamp .- mean(hsamp)).^2
    return sqrt(σ2/n)
end


"""
function MCSEbm(q::GibbsQoI, xsamp::Vector)

Estimates the Monte Carlo standard error (se = std(h(x))/√n) using the batch means (BM) method.

# Arguments
- `q :: GibbsQoI`     : Gibbs QoI struct
- `xsamp :: Vector`   : vector of samples

# Outputs
- `se :: Float64`     : Monte Carlo standard error
"""
function MCSEbm(q::GibbsQoI, xsamp::Vector)
    n = length(xsamp)
    a = Int(floor(sqrt(n)))
    Ȳk = [(1/a)*sum([q.h(xsamp[k*a+i]) for i = 1:a]) for k = 0:(a-1)]
    h̄ = mean(q.h.(xsamp))
    σ2 = a/(a-1) * sum((Ȳk .- h̄).^2)
    return sqrt(σ2/n)
end


"""
function MCSEbm(q::GibbsQoI, xsamp::Vector)

Estimates the Monte Carlo standard error (se = std(h(x))/√n) using the overlapping batch means (OBM) method.

# Arguments
- `q :: GibbsQoI`     : Gibbs QoI struct
- `xsamp :: Vector`   : vector of samples

# Outputs
- `se :: Float64`     : Monte Carlo standard error
"""
function MCSEobm(q::GibbsQoI, xsamp::Vector)
    n = length(xsamp)
    a = Int(floor(sqrt(n)))
    Ȳk = [(1/a)*sum(q.h.(xsamp[(k+1):(k+a)])) for k = 0:(n-a)]
    h̄ = mean(q.h.(xsamp))
    σ2 = n*a/(n-a)/(n-a+1) * sum((Ȳk .- h̄).^2)
    return sqrt(σ2/n)
end


"""
function EffSampleSize(q::GibbsQoI, xsamp::Vector)

Computes the effective sample size of a set of nMC samples from an MCMC chain. 

# Arguments
- `q :: GibbsQoI`     : Gibbs QoI struct
- `xsamp :: Vector`   : nMC-vector of x-samples

# Outputs
- `ESS :: Float64`    : effective sample size (where 1 < ESS < nMC)
"""
function EffSampleSize(q::GibbsQoI, xsamp::Vector)
    hsamp = q.h.(xsamp)
    return EffSampleSize(hsamp)
end


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
    

"""
function ISWeightDiagnostic(wsamp::Vector)

Returns a diagnostic value for importance sampling weights using an unnormalized biasing distribution. The importance sampling estimate is reliable when the diagnostic is less than 5. 
# Arguments
- `wsamp :: Vector`   : nMC-vector of 1-dim. weights

# Outputs
- `diagnostic :: Float64`    : diagnostic value (want < 5)
"""
function ISWeightDiagnostic(wsamp::Vector)
    w̄ = mean(wsamp)
    N = length(wsamp)

    return 1/N * sum( (wsamp ./ w̄ .- 1).^2 )
end