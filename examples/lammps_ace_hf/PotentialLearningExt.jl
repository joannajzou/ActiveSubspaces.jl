function get_all_forces_mag(ds::DataSet)
    return reduce(vcat, [norm.(get_values(get_forces(ds[c]))) for c in 1:length(ds)])
end



function get_all_forces_mag(
    ds::DataSet,
    lb::PotentialLearning.LinearBasisPotential
)
    force_descriptors = [reduce(vcat, get_values(get_force_descriptors(dsi)) ) for dsi in ds]
    force_pred = [lb.β0[1] .+  dB' * lb.β for dB in [reduce(hcat, fi) for fi in force_descriptors]]
    return reduce(vcat, [norm.([f[k:k+2] for k = 1:3:length(f)]) for f in force_pred])
end