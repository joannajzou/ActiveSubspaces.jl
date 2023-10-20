include("lammps_utils.jl")

# prepare LAMMPS input scripts for running MD simulation ----------------------------------------------

# draw samples
nsamp = 5
θsamp = [rand(ρθ) for i = 1:nsamp]
# save to file
JLD.save("$(simdir)coeff_samples.jld",
    "θsamp", θsamp
)

# write IAP coefficients and min configuration to file
for n = 1:nsamp
    println("coeff $n")

    # create directory
    exp_dir = "$(simdir)coeff_$n/"
    run(`mkdir -p $exp_dir`)

    # copy over input script
    run(`cp $(simdir)in.lammps $exp_dir`)

    # write parameters to file
    β_to_file(θsamp[n], ace, exp_dir) 

    # write starting configuration to file
    println("energy min:")
    try
        @time energy_minimization(exp_dir; maxeval=50000)
    catch 
        println("copying from main")
        run(`cp data/starting_configuration.lj $exp_dir`)
    end
end


Temp = 0.6*120          # temperature
Tend = Int(5E6)         # number of steps
dT = 500                # time step output
dt = 0.0025

for n = 1:nsamp
    println("coeff $n")
    exp_dir = "$(simdir)coeff_$n/"
    # run LAMMPS.jl
    try
        @time run_md(Tend, exp_dir, exp_dir; seed=1, dT=dT, dt=dt, Temp=Temp)
        run(`python3 to_extxyz.py $(simdir)coeff_$n/`)
    catch
        println("WARN: run $n failed, skipped")
    end
end
