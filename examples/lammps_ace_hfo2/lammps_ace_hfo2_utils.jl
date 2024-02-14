using LAMMPS


function run_md(
    Tend::Int,
    save_dir::String;
    seed = 1,
    Temp = 0.5,
    Tdamp = 0.1, 
    dt = 0.001,
    dT = 10)

    try
        mkdir(save_dir)
    catch
    end

    d = LMP(["-screen", "none"]) do lmp
        for i = 1
            # command(lmp, "log none")
            command(lmp, "log $(save_dir)log.lammps")
            command(lmp, "units metal")
            command(lmp, "dimension 3")
            command(lmp, "atom_style atomic")
            command(lmp, "atom_modify map array")
            command(lmp, "boundary p p p")

            # Setup box and atoms
            command(lmp, "read_data starting_configuration.lj") # starting_configuration.lj

            command(lmp, "velocity all create $(Temp) $seed mom yes rot yes dist gaussian")
            
            # Setup Forcefield
            command(lmp, "pair_style pace")
            command(lmp, "pair_coeff * * $(save_dir)parameters.ace Hf O")
            
            # computes
            # command(lmp, "compute S all pressure NULL virial") # Stress tensor without kinetic energy component (only the virial)
            # command(lmp, "fix vir all ave/time 1 $dT $dT c_S[1] c_S[2] c_S[3] c_S[6] c_S[5] c_S[4] file $(save_dir)tmp.virial")
            # command(lmp, "compute rdf all rdf 100")      # radial distribution function 
            # command(lmp, "fix frdf all ave/time $dT $(Int(Tend/dT)-1) $Tend c_rdf[*] file $(save_dir)tmp.rdf mode vector")

            command(lmp, "compute pe all pe")                       # potential energy
            command(lmp, "fix fpe all ave/time 1 $dT $dT c_pe file $(save_dir)tmp.pe")
            # command(lmp, "compute pea all pe/atom")                       # potential energy
            
            command(lmp, "compute msd all msd com yes")
            command(lmp, "fix fmsd all ave/time 1 $dT $dT c_msd[4] file $(save_dir)tmp.msd")
            # command(lmp, "compute disp all displace/atom")

            command(lmp, "thermo $dT")
            command(lmp, "fix nvt all nvt temp $Temp $Temp $Tdamp")
            # command(lmp, "fix l0 all nve langevin $Temp $Temp 1 $seed")

            command(lmp, "timestep $dt")
            # command(lmp, "dump 1 all xyz $(dT) $(save_dir)dump.xyz")
            command(lmp, "dump 2 all custom $(dT) $(save_dir)dump.positions_and_forces type id x y z fx fy fz")
            
            command(lmp, "run $Tend")
            command(lmp, "unfix nvt")
            command(lmp, "unfix fpe")
            # command(lmp, "undump 1")
            command(lmp, "undump 2")
            command(lmp, "clear")
        end
    end

end

# Remember to change parameters.ace ACE1.julia -> ACE
function β_to_file(
    β::Vector,
    ace::ACE,
    save_dir::String)
    
    basis = get_rpi(ace)
    IP = JuLIP.MLIPs.combine(basis, β)
    ACE1pack.Export.export_ace("$(save_dir)/parameters.ace", IP)

    # Make adjustment for package name ACE1.jl -> ACE.jl
    fff = readlines("$(save_dir)/parameters.ace")
    fff[10] = "radbasename=ACE.jl.Basic"
    open("$(save_dir)/parameters.ace", "w") do io
        for l in fff
            write(io, l*"\n")
        end
    end
end


function init_exp_dir(
    n::Union{Int,String},
    θ::Vector,
    ace::ACE,
    simdir::String,
    in_script::String,
    min_config::String,
    energy_min=false)

    println("coeff $n")

    # create directory
    exp_dir = "$(simdir)coeff_$n/"
    run(`mkdir -p $exp_dir`)

    # copy over input script
    run(`cp $in_script $exp_dir`)

    # write parameters to file
    β_to_file(θ, ace, exp_dir) 

    # write starting configuration to file
    if energy_min == true
        println("energy min:")
        try
            @time energy_minimization(exp_dir; maxeval=50000)
        catch 
            println("copying from main")
            run(`cp $min_config $exp_dir`)
        end
    else
        run(`cp $min_config $exp_dir`)
    end

    return exp_dir
end

