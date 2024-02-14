include("main.jl")
include("lammps_ace_hf_utils.jl")

# settings for MD simulation --------------------------------------------------------------------------
Temp = 200              # temperature
Tempb = 500
Tend = Int(5E5)         # number of steps
dT = 25                 # time step output
dt = 0.0001


# draw samples
nsamp = 100
θsamp = [rand(ρθ) for i = 1:nsamp]
# save to file
JLD.save("$(simdir)coeff_samples_101-200.jld",
    "θsamp", θsamp
)

# write IAP coefficients and min configuration to file
for n = 1:nsamp
    coeff = n + 100
    println("coeff $coeff")

    # create directory
    exp_dir = "$(simdir)coeff_$coeff/"
    run(`mkdir -p $exp_dir`)

    # copy over input script
    run(`cp run_md_hf.in $exp_dir`)

    # write parameters to file
    β_to_file(θsamp[n], ace, exp_dir) 
    β_to_file(ρθ.μ, ace, exp_dir) 

    # write starting configuration to file
    println("energy min:")
    try
        @time energy_minimization(exp_dir; maxeval=50000)
    catch 
        println("copying from main")
        run(`cp $(simdir)starting_configuration.lj $exp_dir`)
    end

    try
        @time run_md(Tend, exp_dir; seed=1, dT=dT, dt=dt, Temp=Tempb)
        run(`python3 to_extxyz.py $(simdir)coeff_$coeff/`)
    catch
        println("WARN: run $coeff failed, skipped")
    end

end



for n = 1:nsamp
    println("coeff $n")
    exp_dir = "$(simdir)coeff_$n/"
    
    try
        # @time run_md(Tend, exp_dir; seed=1, dT=dT, dt=dt, Temp=Temp)
        run(`python3 to_extxyz.py $(simdir)coeff_$n/`)
    catch
        println("WARN: run $n failed, skipped")
    end
end
