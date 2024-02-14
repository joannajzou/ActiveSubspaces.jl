function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, simdir::String)
# compute gradQ from LAMMPS md runs
    nsamp = length(θsamp)
    ∇Q = Vector{Vector{Float64}}(undef, nsamp)
    ∇Var = Vector{Vector{Float64}}(undef, nsamp)
    id_skip = []
    for j = 1:nsamp
        coeff = j+122
        println("sample $coeff")
        # if !isfile("$(simdir)coeff_$coeff/gradQ_meanenergy.jld")
        try
            ds0 = load_data("$(simdir)coeff_$coeff/data.xyz", ExtXYZ(u"eV", u"Å"))
            ds = ds0[2001:end] # remove transient phase
            Φsamp = sum.(get_values.(compute_local_descriptors(ds, ace)))
            JLD.save("$(simdir)coeff_$coeff/energy_descriptors.jld", "Bsamp", Φsamp)
            MCint = MCSamples(Φsamp)

            # gradient
            ∇Q[j] = grad_expectation(θsamp[j], q, MCint)
            ∇Var[j] = compute_gradvar(θsamp[j], q, MCint)

            # diagnostics
            qoi = GibbsQoI(h = x -> q.h(x, θsamp[j]), p=Gibbs(q.p, θ=θsamp[j]))
            se = MCSEbm(qoi, Φsamp)
            ess = EffSampleSize(qoi, Φsamp)

            # save data
            JLD.save("$(simdir)coeff_$coeff/gradQ_meanenergy.jld",
                "∇Q", ∇Q[j],
                "se", se,
                "ess", ess)
            JLD.save("$(simdir)coeff_$coeff/gradQ_energyvar.jld",
                "∇Q", ∇Var[j],
            )
        catch
            append!(id_skip, coeff)
        end
    end
    return ∇Q, ∇Var, id_skip
end



function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, integrator::ISMixSamples; gradh::Function=nothing)
    # computes gradient of Q using IS integration and returns IS diagnostics
        nsamp = length(θsamp)
        ∇Qis, his, wis, wts = init_IS_arrays_mix(nsamp)
        metrics = init_metrics_dict(nsamp)
    
        for k = 1:nsamp
            t = @elapsed ∇Qis[k], his[k], wis[k], wts[k] = grad_expectation(θsamp[k], q, integrator; gradh=gradh)
            println("samp $k: $t sec")
        end
    
        nMC = length(his[1]) 
        w̃is = [[his[i][j] * wis[i][j] for j = 1:nMC] for i = 1:nsamp]
        metrics["θsamp"] = θsamp
        metrics["wvar"] = [norm(ISWeightVariance(wi)) for wi in w̃is]
        metrics["wESS"] = [ISWeightESS(wi)/nMC for wi in w̃is]
        metrics["wdiag"] = [ISWeightDiagnostic(wi) for wi in wis]
        metrics["mixwts"] = wts
    
        return ∇Qis, metrics
    end


function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, integrator::ISIntegrator; gradh::Function=nothing)
# computes gradient of Q using IS integration and returns IS diagnostics
    nsamp = length(θsamp)
    ∇Qis, his, wis = init_IS_arrays(nsamp)
    metrics = init_metrics_dict(nsamp)

    for k = 1:nsamp
        t = @elapsed ∇Qis[k], his[k], wis[k] = grad_expectation(θsamp[k], q, integrator; gradh=gradh)
        println("samp $k: $t sec")
    end

    nMC = length(his[1]) 
    w̃is = [[his[i][j] * wis[i][j] for j = 1:nMC] for i = 1:nsamp]
    metrics["θsamp"] = θsamp
    metrics["wvar"] = [norm(ISWeightVariance(wi)) for wi in w̃is]
    metrics["wESS"] = [ISWeightESS(wi)/nMC for wi in w̃is]
    metrics["wdiag"] = [ISWeightDiagnostic(wi) for wi in wis]

    return ∇Qis, metrics
end


function compute_gradvar(θ::Vector{<:Real},
    q::GibbsQoI,
    integrator::Integrator)

    q2 = GibbsQoI(h=(x, γ) -> q.h(x, γ)^2,
        ∇h=(x, γ) -> 2*q.h(x, γ)*q.∇h(x, γ),
        p = q.p)

    EX = expectation(θ, q, integrator)
    ∇EX = grad_expectation(θ, q, integrator)
    ∇EX2 = grad_expectation(θ, q2, integrator)

    #Var[X] = E[X^2] - E[X]^2
    return ∇EX2 - 2*EX.*∇EX
end


function init_IS_arrays(nsamp)
# initialize arrays for compute_gradQ() with IS integration
    ∇Q_arr = Vector{Vector{Float64}}(undef, nsamp)
    ∇h_arr = Vector{Vector}(undef, nsamp)
    w_arr = Vector{Vector{Float64}}(undef, nsamp)
    return ∇Q_arr, ∇h_arr, w_arr
end

function init_IS_arrays_mix(nsamp)
# initialize arrays for compute_gradQ() with IS integration
    ∇Q_arr = Vector{Vector{Float64}}(undef, nsamp)
    ∇h_arr = Vector{Vector}(undef, nsamp)
    w_arr = Vector{Vector{Float64}}(undef, nsamp)
    wts_arr = Vector{Vector{Float64}}(undef, nsamp)
    return ∇Q_arr, ∇h_arr, w_arr, wts_arr
end

function combine_exp(
    dir1::String,
    dir2::String,
    filename::String,
    qname::String,
)

    q1 = JLD.load(dir1*filename)[qname]
    q2 = JLD.load(dir2*filename)[qname]
    q = reduce(vcat, (q1, q2))
    return q 
end

function draw_samples_as_mode(W::Matrix, ind::Int64, ρθ::Distribution, ny::Int64)
    dims = Vector(1:size(W,1))
    W1 = W[:,ind:ind] # type as matrix
    W2 = W[:, dims[Not(ind)]]

    # compute sampling density
    π_y = compute_marginal(W1, ρθ)
    π_z = compute_marginal(W2, ρθ)

    # draw samples
    ysamp = [rand(π_y) for i = 1:ny]
    θy = [W1*y + W2*π_z.μ for y in ysamp]

    return θy
end

function compute_quantiles(Qsamp; quantiles=0.1:0.1:0.9)
    return [quantile!(Qsamp, q) for q in quantiles]
end