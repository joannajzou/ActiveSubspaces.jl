"""
struct MDSim <: Sampler

Defines the struct with parameters for sampling by molecular dynamics (MD) simulation in LAMMPS.

# Arguments
- `L :: Int`        : time length between evaluating accept/reject criterion
- `ϵ :: Real`       : step size

"""
struct MDSim <: Sampler
    iap :: InteratomicPotential     # interatomic potential defining energies and forces
    temp :: Float64                 # temperature of simulation
    T :: Int                        # final time of simulation
    dt :: Real                      # time step of simulation
    Nout :: Int                     # step size for printing output
end


function sample(coeff :: Vector{Float64},   # vector of IAP coefficients
            emin_fun :: Function,           # energy minimization function
            md_fun :: Function,             # MD simulation function
            md :: MDSim,                    # MD simulation parameters
            simdir :: String)               # directory for saving simulation results
    
    # write coefficient to file
    β_to_file(β, iap, simdir) 
    
    # find minimum energy configuration for starting point 
    emin_fun(simdir)
    
    # run MD simulation
    md_fun(simdir, md)
end

