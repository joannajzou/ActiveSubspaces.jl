"""
    Discrepancy

    A struct of abstract type Discrepancy is function that computes the discrepancy between two probability densities.
"""
abstract type Discrepancy end


struct MaxMeanDiscrepancy <: Discrepancy
    k::Kernel
end


struct KernelSteinDiscrepancy <: Discrepancy
    k::Kernel
end

function compute_discrepancy(
    sp::Function,
    sq::Function,
    qsamp::Vector,
    d::KernelSteinDiscrepancy
    )

    δ(x) = sp(x) - sq(x)

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
    qsamp::Vector,
    d::Discrepancy
    )

    sp(x) = gradlogpdf(p, x)
    sq(x) = gradlogpdf(q, x)
    return compute_discrepancy(sp, sq, qsamp, d)
end

function compute_discrepancy(
    sp::Function,
    q::Distribution,
    qsamp::Vector,
    d::Discrepancy
    )

    sq(x) = gradlogpdf(q, x)
    return compute_discrepancy(sp, sq, qsamp, d)
end

