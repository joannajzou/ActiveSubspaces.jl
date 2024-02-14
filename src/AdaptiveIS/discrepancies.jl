using LinearAlgebra
"""
    Discrepancy

    A struct of abstract type Discrepancy is function that computes the discrepancy between two probability densities.
"""
abstract type Discrepancy end



struct MaxMeanDiscrepancy <: Discrepancy
    k::Kernel
end

"""
    KernelSteinDiscrepancy <: Discrepancy
"""
mutable struct KernelSteinDiscrepancy <: Discrepancy
    k::Kernel
    qsamp::Vector
end

function compute_discrepancy(
    sp::Function,
    sq::Function,
    d::KernelSteinDiscrepancy
)
    δ(x) = sp(x) - sq(x)
    
    qsamp = d.qsamp
    nq = length(qsamp)
    if nq > 2000
        ids = StatsBase.sample(1:nq, 2000; replace=false)
        qsamp1 = qsamp[ids[1:1000]]
        qsamp2 = qsamp[ids[1001:end]]
    else
        qsamp1 = qsamp
        qsamp2 = qsamp
    end
    return sum([δ(x)*compute_kernel(x,x̃,d.k)*δ(x̃) for x in qsamp1, x̃ in qsamp2])
    
end

function compute_discrepancy(
    p::Distribution,
    q::Distribution,
    d::KernelSteinDiscrepancy
)
    sp(x) = gradlogpdf(p, x)
    sq(x) = gradlogpdf(q, x)
    return compute_discrepancy(sp, sq, d)
end

function compute_discrepancy(
    sp::Function,
    q::Distribution,
    d::KernelSteinDiscrepancy
)
    sq(x) = gradlogpdf(q, x)
    return compute_discrepancy(sp, sq, d)
end

function compute_discrepancy(
    p::Distribution,
    sq::Function,
    d::KernelSteinDiscrepancy
)
    sp(x) = gradlogpdf(p, x)
    return compute_discrepancy(sp, sq, d)
end


"""
    KLDivergence <: Discrepancy
"""
struct KLDivergence <: Discrepancy
    int::QuadIntegrator
end


function compute_discrepancy(
    p::Gibbs,
    q::Gibbs,
    d::KLDivergence,
)
    Zp = normconst(p, d.int)
    Zq = normconst(q, d.int)

    h(x) = updf(p, x)/Zp .* ((logupdf(p, x) - log(Zp)) - (logupdf(q, x) - log(Zq)))
    return sum(d.int.w .* h.(d.int.ξ))

end


"""
    FisherDivergence <: Discrepancy
"""
struct FisherDivergence <: Discrepancy
    int::Integrator
end

function compute_discrepancy(
    p::Gibbs,
    q::Gibbs,
    d::FisherDivergence,
)
    sp(x) = gradlogpdf(p, x)
    sq(x) = gradlogpdf(q, x)

    if typeof(d.int) <: QuadIntegrator
        Zp = normconst(p, d.int)
        h(x) = updf(p, x)/Zp .* norm(sp(x) - sq(x))^2
        return sum(d.int.w .* h.(d.int.ξ))
    
    elseif typeof(d.int) <: MCMC
        xsamp = rand(p, d.int.n, d.int.sampler, d.int.ρ0) 
        h = x -> norm(sp(x) - sq(x))^2
        return sum(h.(xsamp)) / length(xsamp)

    elseif typeof(d.int) <: MCSamples
        h = x -> norm(sp(x) - sq(x))^2
        return sum(h.(d.int.xsamp)) / length(d.int.xsamp)
        
    end
end