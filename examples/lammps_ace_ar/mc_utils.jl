function compute_Q(θsamp::Vector{Vector{Float64}}, ids::Vector, q::GibbsQoI, simdir::String)
    nsamp = length(θsamp)
    Q = zeros(nsamp)
    for j = 1:nsamp
        coeff = ids[j]
        println("sample $coeff")
        Φsamp = JLD.load("$(simdir)coeff_$coeff/energy_descriptors.jld")["Bsamp"]
        MCint = MCSamples(Φsamp)
        Q[j] = expectation(θsamp[j], q, MCint)
        JLD.save("$(simdir)coeff_$coeff/Q_meanenergy.jld", "Q", Q[j])

    end
    return Q
end


function compute_gradQ(θsamp::Vector{Vector{Float64}}, q::GibbsQoI, simdir::String)
# compute gradQ from LAMMPS md runs
    nsamp = length(θsamp)
    ∇Q = Vector{Vector{Float64}}(undef, nsamp)
    id_skip = []
    for j = 1:nsamp
        coeff = j+2063
        println("sample $coeff")
        # if !isfile("$(simdir)coeff_$coeff/gradQ_meanenergy.jld")
        try
            ds = load_data("$(simdir)coeff_$coeff/data.xyz", ExtXYZ(u"eV", u"Å"))
            e_descr = compute_local_descriptors(ds, ace)
            Φsamp = sum.(get_values.(e_descr))
            MCint = MCSamples(Φsamp)

            # gradient
            ∇Q[j] = grad_expectation(θsamp[j], q, MCint)

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

function intersection_indices(arr1, arr2)
    indices = Int[]

    for (index, value) in enumerate(arr1)
        id2 = findall(x -> x == value, arr2)
        if !isempty(id2)
            push!(indices, id2[1])
        end
    end

    return indices
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