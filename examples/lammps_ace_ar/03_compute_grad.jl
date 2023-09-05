include("00_spec_model.jl")
include("qoi_meanenergy.jl")
include("mc_utils.jl")
# include("plotting_utils.jl")

Temp = 72.0
Temp_b = 150.0        # higher temp for biasing dist.    
Tempscl = Temp / Temp_b
biasdir = "$(simdir)Temp_$(Temp_b)/"


# load data #####################################################################################
# coeff distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)
βdim = length(μ)

# coeff samples
βsamp = JLD.load("$(simdir)coeff_samples_aug.jld")["βsamp"]
nsamp = length(βsamp)

# biasing distribution samples
λsamp = JLD.load("$(biasdir)coeff_samples.jld")["βsamp"]
nλ = length(λsamp)

βaug = reduce(vcat, (βsamp_2, λsamp_2))
JLD.save("$(simdir)coeff_samples_aug.jld", "βsamp", βaug)


# Monte Carlo - standard ########################################################################
∇Qmc, idskip = compute_gradQ(λsamp_2, q, simdir; gradh=∇h) 
∇Qmc, idskip = compute_gradQ(βsamp_c, q, biasdir; gradh=∇h) 
# JLD.save("$(biasdir)coeff_skip.jld", "id_skip", idskip)

# remove skipped indices
idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]
idkeep_β = symdiff(1:nsamp, idskip)
βsamp_2 = βsamp[idkeep_β]

idskip = JLD.load("$(biasdir)coeff_skip.jld")["id_skip"]
idkeep = symdiff(1:nλ, idskip)
λsamp_2 = λsamp[idkeep]


# importance sampling - single distribution #####################################################
# πg = Gibbs(πgibbs, θ=πβ.μ)

# # compute descriptors 
# Φsamp = JLD.load("$(simdir)coeff_mean/energy_descriptors.jld")["Bsamp"]

# # define integrator
# ISint = ISSamples(πg, Φsamp)

# # compute
# ∇Qis, metrics = compute_gradQ(βsamp_2, q, ISint; gradh=∇h)

# JLD.save("$(simdir)gradQ_meanenergy_IS_single.jld",
#     "∇Q", ∇Qis,
#     "metrics", metrics)


# importance sampling - mixture distribution, equal weights #############################################
# randomly sample component PDFs

nprop_arr = [50, 100, 150]
nprop_tot = nprop_arr[end]
nMC = 10000
knl = RBF(Euclidean(βdim); ℓ=1e-8) # RBF(Euclidean(inv(Σ)); ℓ=1)



for iter = 1:4
    println("======================== ITER $iter ========================")
    t = @elapsed begin
        centers = JLD.load("$(simdir)/gradQ_IS_EW_nc=150/gradQ_ISM_Temp_$(iter+4).jld")["centers"]
        center_ids = intersection_indices(centers, βsamp_c)
        # center_ids = rand(idkeep, nprop_tot)

        # load from file
        samp_all = Vector{Vector}(undef, nprop_tot)
        for i = 1:nprop_tot
            num = center_ids[i]+500
            Φsamp = JLD.load("$(biasdir)coeff_$num/energy_descriptors.jld")["Bsamp"]
            samp_all[i] = Φsamp
        end
    end
    println("overhead: $t sec.")

    for nprop in nprop_arr
        println("----------- nprop $nprop -----------")

        centers = βsamp_c[center_ids[1:nprop]]
        samps = samp_all[1:nprop]

        # define mixture model
        πgibbs2 = Gibbs(πgibbs, β=Tempscl)
        πg = [Gibbs(πgibbs2, θ=c) for c in centers]
        mm = MixtureModel(πg)

        # draw fixed set of samples
        Φsamp_mix, catratio = rand(mm, nMC, samps)
        ISint = ISSamples(mm, Φsamp_mix)
        
        
        # compute with equal mixture weights
        println("COMPUTE GRAD Q - EQUAL WTS")
        ∇Qis, metrics = compute_gradQ(βsamp_2, q, ISint; gradh=∇h)

        JLD.save("$(simdir)/gradQ_IS_EW_nc=$(nprop)/gradQ_ISM_Temp_$(iter).jld",
            "∇Q", ∇Qis,
            "centers", centers,
            "metrics", metrics)

        # compute with variable mixture weights
        ISmix = ISMixSamples(mm, nMC, knl, samps)

        println("COMPUTE GRAD Q - VAR WTS")
        ∇Qis_m, metrics_m = compute_gradQ(βaug[1:1], q, ISmix; gradh=∇h)

        JLD.save("$(simdir)/gradQ_IS_VW_nc=$(nprop)/gradQ_ISM_Temp_$(iter).jld",
            "∇Q", ∇Qis_m,
            "centers", centers,
            "metrics", metrics_m)

    end
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
