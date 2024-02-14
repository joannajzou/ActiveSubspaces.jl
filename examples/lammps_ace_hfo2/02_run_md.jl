include("lammps_ace_hfo2_utils.jl")

# settings for MD simulation --------------------------------------------------------------------------
Temp = 2000             # temperature
Tend = 5000 # Int(1E5)         # number of steps
dT = 50                # time step output
dt = 0.001


# run MD simulation at mean value ----------------------------------------------
coeff = "mean"
exp_dir = init_exp_dir(coeff, μθ, ace, simdir, "run_md_hfo2.in", "starting_configuration.lj")

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
    exp_dir = "$(simdir)coeff_$n/"
    # run LAMMPS.jl
    try
        @time run_md(Tend, exp_dir; seed=1, dT=dT, dt=dt, Temp=Temp)
        run(`python3 to_extxyz.py $(simdir)coeff_$n/`)
    catch
        println("WARN: run $n failed, skipped")
    end
end
