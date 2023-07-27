function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, simdir::String; gradh::Function=nothing)
# compute gradQ from LAMMPS md runs
    nsamp = length(θsamp)
    ∇Q = Vector{Vector{Float64}}(undef, nsamp)
    id_skip = []
    for j = 1:nsamp
        coeff = j
        println("sample $coeff")
        # if !isfile("$(simdir)coeff_$coeff/gradQ_meanenergy.jld")
        try
            ds = load_data("$(simdir)coeff_$coeff/data.xyz", ExtXYZ(u"eV", u"Å"))
            e_descr = compute_local_descriptors(ds, ace)
            Φsamp = sum.(get_values.(e_descr))
            MCint = MCSamples(Φsamp)

            # gradient
            ∇Q[j] = grad_expectation(θsamp[j], q, MCint, gradh=gradh)

            # diagnostics
            qoi = GibbsQoI(h = x -> q.h(x, θsamp[j]), p=Gibbs(q.p, θ=θsamp[j]))
            se = MCSEbm(qoi, Φsamp)
            ess = EffSampleSize(qoi, Φsamp)

            # save data
            JLD.save("$(simdir)coeff_$coeff/energy_descriptors.jld", "Bsamp", Φsamp)
            JLD.save("$(simdir)coeff_$coeff/gradQ_meanenergy.jld",
                    "∇Q", ∇Q[j],
                    "se", se,
                    "ess", ess)
        catch
            append!(id_skip, coeff)
        end
    end
    return ∇Q, id_skip
end


function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, integrator::ISIntegrator, simdir::String, ids::Vector; gradh::Function=nothing)
# computes gradient of Q using IS integration and returns IS diagnostics
    nsamp = length(θsamp)
    ∇Qis, his, wis = init_IS_arrays(nsamp)
    metrics = init_metrics_dict(nsamp)

    for k = 1:nsamp
        coeff = ids[k]
        t = @elapsed ∇Qis[k], his[k], wis[k] = grad_expectation(θsamp[k], q, integrator; gradh=gradh)
        println("samp $coeff: $t sec")
        JLD.save("$(simdir)coeff_$coeff/gradQ_meanenergy_IS.jld",
                "∇Qis", ∇Qis[k])
    end

    nMC = length(his[1]) 
    w̃is = [[his[i][j] * wis[i][j] for j = 1:nMC] for i = 1:nsamp]
    metrics["θsamp"] = θsamp
    metrics["wvar"] = [norm(ISWeightVariance(wi)) for wi in w̃is]
    metrics["wESS"] = [ISWeightESS(wi)/nMC for wi in w̃is]
    metrics["wdiag"] = [ISWeightDiagnostic(wi) for wi in wis]

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


function init_IS_arrays(nsamp)
# initialize arrays for compute_gradQ() with IS integration
    ∇Q_arr = Vector{Vector{Float64}}(undef, nsamp)
    ∇h_arr = Vector{Vector}(undef, nsamp)
    w_arr = Vector{Vector{Float64}}(undef, nsamp)
    return ∇Q_arr, ∇h_arr, w_arr
end


