include("define_md_LAMMPS_ace.jl")


## Run MD for different coefficients and different seeds
Tend = Int(0.5E6)
dT = 50
Temp = 0.65*120
file_dir = "./TEMP/"
for coeff = 1:20
    println("coeff $coeff")
    save_dir = "ACE_MD/coeff_$coeff/"
    run(`mkdir -p $save_dir`)
    # for seed = 1:100
        @time run_md(Tend, file_dir, save_dir; seed = 1, dT = dT, dt = 0.005, Temp = Temp)
        run(`python3 to_extxyz.py ACE_MD/coeff_$(coeff)/`)
    # end
end
