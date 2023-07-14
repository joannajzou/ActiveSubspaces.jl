function compute_covmatrix_ref(quadpts::Tuple, q::GibbsQoI, p::Distribution, integrator::QuadIntegrator)
# computes 2x2 reference covariance matrix using quadrature
    ξθ, wθ = quadpts
    ngrid = length(ξθ)

    Cref = zeros(2,2)
    for i = 1:ngrid
        for j = 1:ngrid
            println("($i, $j)")
            ξij = [ξθ[i], ξθ[j]]
            ∇Qij = grad_expectation(ξij, q, integrator)
            Cref .+= ∇Qij*∇Qij' * pdf(p, ξij) * wθ[i] * wθ[j]
        end
    end

    return Cref
end


function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, integrator::MCIntegrator; gradh::Function=nothing)
# computes gradient of Q using MC integration
    ∇Qmc = map(θ -> grad_expectation(θ, q, integrator, gradh=gradh), θsamp)
    return ∇Qmc
end


function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, integrator::ISIntegrator; gradh::Function=nothing)
# computes gradient of Q using IS integration and returns IS diagnostics
    nsamp = length(θsamp)
    ∇Qis, his, wis = init_IS_arrays(nsamp)
    metrics = init_metrics_dict(nsamp)

    for k = 1:nsamp
        println("samp $k: $t sec.")
    end

    nMC = length(his[1]) 
    w̃is = [[his[i][j] * wis[i][j] for j = 1:nMC] for i = 1:nsamp]
    metrics["θsamp"] = θsamp
    metrics["wvar"] = [norm(ISWeightVariance(wi)) for wi in w̃is]
    metrics["wESS"] = [ISWeightESS(wi)/nMC for wi in w̃is]
    metrics["wdiag"] = [ISWeightDiagnostic(wi) for wi in wis]

    return ∇Qis, metrics
end


function init_IS_arrays(nsamp)
# initialize arrays for compute_gradQ() with IS integration
    ∇Q_arr = Vector{Vector{Float64}}(undef, nsamp)
    ∇h_arr = Vector{Vector}(undef, nsamp)
    w_arr = Vector{Vector{Float64}}(undef, nsamp)
    return ∇Q_arr, ∇h_arr, w_arr
end


function remove_outliers(θsamp)
    θmat = reduce(hcat, θsamp)

    ids1 = findall(x -> x <= 6.0, θmat[1,:])
    ids2 = findall(x -> x >= 2.0, θmat[2,:])
    ids = findall(x -> x in ids1, ids2)
    return θsamp[ids]
end