include("00_spec_model.jl")
include("define_md_LAMMPS_ace.jl")

# load posterior distribution
μ = JLD.load("$(simdir)coeff_distribution.jld")["μ"]
Σ = JLD.load("$(simdir)coeff_distribution.jld")["Σ"]
πβ = MvNormal(μ, Σ)


# function ############################################################################################
function run_md_ar(coeff, simdir::String, β::Vector{Float64}, Temp::Float64)
    Tend = Int(0.5E6)       # number of steps
    dT = 50                 # time step output

    println("coeff $coeff")

    # create directory
    save_dir = "$(simdir)coeff_$coeff/"
    run(`mkdir -p $save_dir`)

    # write coefficients to file (parameters.ace)
    β_to_file(β, ace, save_dir) 

    # for seed = 1:100
    # @time energy_minimization(save_dir; maxeval=50000)
    @time run_md(Tend, save_dir, save_dir; seed = 1, dT = dT, dt = 0.00005, Temp = Temp)
    run(`python3 to_extxyz.py $(simdir)coeff_$(coeff)/`)

end


# compute energy with MD #############################################################################
nsamp = 2000            # number of coefficient samples
Temp = 0.6*120          # temperature
βsamp = [rand(πβ) for i = 1:nsamp]
JLD.save("$(simdir)coeff_samples.jld",
        "βsamp", βsamp)


# compute at mean/nominal get_values
run_md_ar("nom", simdir, πβ.μ, 0.7*120)


# iterate over coefficient samples
for coeff = 1:nsamp       
    try
        run_md_ar(coeff, simdir, βsamp[coeff], Temp)
    catch
        println("WARN: run $coeff failed, skipped")
    end
end



    


