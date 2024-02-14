include("subsampling_utils.jl")
include("plotting_utils.jl")
include("PotentialLearningExt.jl")

# define ACE basis --------------------------------------------------------------------------------------
elname = "Hf"
nbody = 5
deg = 6

ace = ACE(species = [:Hf],          # species
          body_order = nbody,           # n-body
          polynomial_degree = deg,      # degree of polynomials
          wL = 1.0,                     # Defaults, See ACE.jl documentation 
          csp = 1.0,                    # Defaults, See ACE.jl documentation 
          r0 = 1.0,                     # minimum distance between atoms
          rcutoff = 10.0)                # cutoff radius 


# load dataset -------------------------------------------------------------------
file_arr = readext("data/", "xyz")
nfile = length(file_arr)
confs_arr = [load_data("data/"*file, ExtXYZ(u"eV", u"Å")) for file in file_arr]
confs = concat_dataset(confs_arr)

# id of configurations per file
n = 0
confs_id = Vector{Vector{Int64}}(undef, nfile)
for k = 1:nfile
    global n
    confs_id[k] = (n+1):(n+length(confs_arr[k]))
    n += length(confs_arr[k])
end


# Update dataset by adding energy (local) descriptors ------------------------------
println("Computing local descriptors")
# e_descr = compute_local_descriptors(confs, ace)
# f_descr = compute_force_descriptors(confs, ace)

# JLD.save("data/global_energy_descriptors_$(nbody)body_$(deg)deg.jld", 
#     "e_descr", e_descr,
# )
# JLD.save("data/global_force_descriptors_$(nbody)body_$(deg)deg.jld", 
#     "f_descr", f_descr,
# )

# or load
e_descr = JLD.load("data/$(elname)_energy_descriptors_$nbody-$deg.jld")["e_descr"]
f_descr = JLD.load("data/$(elname)_force_descriptors_$nbody-$deg.jld")["f_descr"]

ds = DataSet(confs .+ e_descr + f_descr)
ndata = length(ds)


# learn using DPP samples ------------------------------------------------------------
ind_all = rand(1:ndata, ndata)
ds_train = ds[ind_all[1:14000]]
ds_test = ds[ind_all[14001:end]]
lb1, _, dpp_inds = train_potential(ds_train, ace, 250)
lb2, _, dpp_inds = train_potential(ds_train, ace, 4000)


JLD.save("data/fitted_params_$(nbody)body_$(deg)deg.jld",
    "dpp_inds", dpp_inds,
    "β", lp.β,
    "Σ", lp.Σ)


# evaluate fit ----------------------------------------------------------------------------------------

# compute error metrics
dict1 = evaluate_fit(ds_train, ds_test, lb1)
dict2 = evaluate_fit(ds_train, ds_test, lb2)



# plot energy and error 
plot_energy([dict2, dict1], "e", "train", ["no. data: 4000", "no. data: 250"])
plot_energy([dict2, dict1], "e", "test", ["no. data: 4000", "no. data: 250"])
plot_energy([dict2, dict1], "f", "train", ["no. data: 4000", "no. data: 250"])
plot_energy([dict2, dict1], "f", "test", ["no. data: 4000", "no. data: 250"])


plot_energy(e_train_pred[Not(621)], e_train[Not(621)])
plot_energy(e_test_pred[Not(4928)], e_test[Not(4928)])

plot_energy(f_train_pred, f_train)
plot_energy(f_test_pred, f_test)
plot_energy(f_test_pred[Not([51736, 51737, 29311, 29312])], f_test[Not([51736, 51737, 29311, 29312])])
plot_error(e_train_err[Not(621)])
plot_error(e_test_err)


type = f 
set = "train"

r0 = minimum([minimum(dict["$(type)_$(set)"]) for dict in dicts])
    r1 = maximum([maximum(dict["$(type)_$(set)"]) for dict in dicts])
    rs = (r1-r0)/10
    colors=["viridis", "orange"]

    fig = Figure(resolution=(500,500))
    if type == "e"
        ax = Axis(fig[1,1], xlabel="E ref. [eV/atom]", ylabel="E pred. [eV/atom]")
    elseif type == "f"
        ax = Axis(fig[1,1], xlabel="F ref. [eV/Å]", ylabel="F pred. [eV/Å]") 
    end

    for (i,d) in enumerate(dicts)
        scatter!(ax, d["$(type)_$(set)"], d["$(type)_$(set)_pred"], markersize=7, label=lbls[i], colormap=(colors[i], 0.25))
    end
    lines!(ax, r0:rs:r1, r0:rs:r1, color=:red)
    axislegend(ax, position = :lt)
    return fig
