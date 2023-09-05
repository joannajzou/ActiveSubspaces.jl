include("00_spec_model.jl")
include("define_md_LAMMPS_ace.jl")
using CairoMakie 

# load posterior distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)


# function ############################################################################################
function run_md_ar(coeff, simdir::String, β::Vector{Float64}, Temp::Real)
    Tend = Int(5E6)       # number of steps
    dT = 500              # time step output

    println("coeff $coeff")

    # create directory
    save_dir = "$(simdir)coeff_$coeff/"
    run(`mkdir -p $save_dir`)

    # write coefficients to file (parameters.ace)
    β_to_file(β, ace, save_dir) 

    # for seed = 1:100
    println("energy min:")
    try
        @time energy_minimization(save_dir; maxeval=50000)
    catch 
        println("copying from main")
        run(`cp starting_configuration.lj $save_dir`)
    end
    println("run MD:")
    @time run_md(Tend, save_dir, save_dir; seed = 1, dT = dT, dt = 0.0025, Temp = Temp)
    run(`python3 to_extxyz.py $(simdir)coeff_$(coeff)/`)
end


# compute energy with MD #############################################################################
nsamp = 500             # number of coefficient samples
Temp = 0.6*120          # temperature

Temp_b = 150.0        # higher temp for biasing dist.    
biasdir = "$(simdir)Temp_$(Temp_b)/"

# βsamp = [rand(πβ) for i = 1:nsamp]

# λsamp = JLD.load("$(biasdir)coeff_samples.jld")["βsamp"]
# nλ = length(λsamp)
# idskip = JLD.load("$(biasdir)coeff_skip.jld")["id_skip"]
# idkeep = symdiff(1:nλ, idskip)
# λsamp_2 = λsamp[idkeep]
# nsamp = length(λsamp_2)

βsamp = JLD.load("$(simdir)coeff_samples.jld")["βsamp"]
nsamp = length(βsamp)
idskip = JLD.load("$(simdir)coeff_skip.jld")["id_skip"]
idkeep_β = symdiff(1:nsamp, idskip)
βsamp_2 = βsamp[idkeep_β]

center_ids = Vector{Int64}()
for iter = 1:4
    centers = JLD.load("$(simdir)/gradQ_IS_EW_nc=150/gradQ_ISM_$(iter).jld")["centers"]
    ids = intersection_indices(centers, βsamp_2)
    center_ids = reduce(vcat, (center_ids, ids))
end
center_ids_fin = unique(center_ids)
βsamp_c = βsamp_2[center_ids_fin]

# JLD.save("$(biasdir)coeff_samples.jld",
#         "βsamp", βsamp)


# compute at mean/nominal get_values
# run_md_ar("mean", biasdir, πβ.μ, Temp_b)

# iterate over coefficient samples
for coeff = 1:nsamp       
    try
        run_md_ar(coeff+500, biasdir, βsamp_c[coeff], Temp_b)
    catch
        println("WARN: run $coeff failed, skipped")
    end
end






    


