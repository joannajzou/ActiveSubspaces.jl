using StatsBase

function plot_val_metric(val_type::String, val_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String; logscl=true)
    colors = [:mediumpurple4, :skyblue1, :seagreen, :goldenrod1, :darkorange2] 
    markers = [:utriangle, :hexagon, :circle, :diamond, :rect]
    nsamp_arr = collect(keys(val_tup[1][val_type]))
    N = length(nsamp_arr)

    fig = Figure(resolution=(900, 600))
    if logscl == true
        ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel=ylab, xticks=(nsamp_arr, [string(nsamp) for nsamp in nsamp_arr]),
        title=ttl, xscale=log10, yscale=log10)   
    else
        ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel=ylab, xticks=(nsamp_arr, [string(nsamp) for nsamp in nsamp_arr]),
        title=ttl, xscale=log10, yticks=0.0:0.005:0.025) 
    end
    offset_ratio = [-0.08, -0.04, 0, 0.04, 0.08]
    # offsets = Point[(-20, 0), (-10, 0), (0,0), (10,0), (20,0)]
    # offsets = [exp.(os) for os in offsets]
    for (i, val_i, lab_i) in zip(1:length(val_tup), val_tup, lab_tup)
        val_med = [median(val_i[val_type][k]) for k in nsamp_arr]
        val_hi = [percentile(val_i[val_type][k], 75) for k in nsamp_arr] # .+ 1e-6
        val_lo = [percentile(val_i[val_type][k], 25) for k in nsamp_arr] # .+ 1e-6
        offsets = offset_ratio[i]*nsamp_arr

        for k = 1:N
            nsamp = nsamp_arr[k]
            if k == 1
                scatter!(ax, [nsamp + offsets[k]], [val_med[k]], color=colors[i], marker=markers[i], markersize=15, label=lab_i)
            else
                scatter!(ax, [nsamp + offsets[k]], [val_med[k]], color=colors[i], marker=markers[i], markersize=15)
            end
            rangebars!(ax, [nsamp + offsets[k]], [val_lo[k]], [val_hi[k]], color=colors[i], linewidth=2, whiskerwidth=2e1 * nsamp)
        end
    end
    fig[1, 2] = Legend(fig, ax, framevisible = false)
    # axislegend(ax, position=:ct)
    fig
end


function plot_IS_diagnostic(met_type::String, met_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String; logscl=false)
    w_arr = [met_i[met_type] for met_i in met_tup]
    nIS = length(w_arr)

    cmin = minimum([minimum(wi) for wi in w_arr])       # minimum value for colorbar
    cmax = maximum([maximum(wi) for wi in w_arr])       # maximum value for colorbar

    θsamp = met_tup[1]["θsamp"]
    θmat = reduce(hcat, θsamp)

    if logscl == true
        cmin = log.(cmin)
        cmax = log.(cmax)
        w_arr = [log.(wi) for wi in w_arr]
    end

    # initialize figure
    fig = Figure(resolution=(450*nIS, 500)) # Figure(resolution=(1600, 1300))
    ax = Vector{Axis}(undef, nIS)
    
    for i = 1:nIS
        wi = w_arr[i]
        
        ax[i] = Axis(fig[1,i][1,1], xlabel="θ₁", ylabel="θ₂",
        title=lab_tup[i]) 
        # plot 2D scatterplot and colorbar
        sc = scatter!(ax[i], θmat[1,:], θmat[2,:], markersize=7, color=wi, colormap=Reverse(:Blues), colorrange=(cmin,cmax)) 
        if i == nIS
            Colorbar(fig[1,i][1,2], sc, label=ylab)
        end  
    end
    Label(fig[0,:], text=ttl, fontsize=20)
    return fig
end


function plot_IS_samples(πg0::Gibbs, met_type::String, met_tup::Tuple, lab_tup::Tuple, dist::Tuple, ttl::String, xlim::Vector, quadpts::Tuple; rev=false, nsamp=100, thresh=:none)
    w_arr = [met_i[met_type] for met_i in met_tup]
    θsamp = met_tup[1]["θsamp"]
    nIS = length(w_arr)

    # initialize figure
    fig = Figure(resolution = (450*nIS, 500))
    ax = Vector{Axis}(undef, nIS)

    for j = 1:nIS
        wi = w_arr[j]
        # sort elements
        if thresh == :none
            if rev == true
                ids = reverse(sortperm(wi))[1:nsamp]
            else # not reversed
                ids = sortperm(wi)[1:nsamp]
            end
        else
            if rev == true
                ids = findall(x -> x > thresh, wi)
            else
                ids = findall(x -> x <= thresh, wi)
            end
        end
        m = length(ids)
        propsamp = round((m/length(θsamp))*100, digits=1)

        ax[j] = Axis(fig[1,j],  xlabel="x", ylabel="pdf(x)", limits=(xlim[1], xlim[2], -0.02, 0.75), title=lab_tup[j])
        xplot = LinRange(xlim[1], xlim[2], 1000)

        # plot first nsamp samples 
        if m != 0
            for (i, θi) in zip(1:m, θsamp[ids])
                πg = Gibbs(πg0, β=1.0, θ=θi)
                if i == 1
                    lines!(ax[j], xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:steelblue, 0.1)) # label="prop: $(propsamp)%") 
                else
                    lines!(ax[j], xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:steelblue, 0.1)) 
                end
            end
        else
            lines!(ax[j], xplot, zeros(length(xplot)), color=(:steelblue, 0.001), label="prop: 0%") 
        end
        # plot biasing distribution
        try
        lines!(ax[j], xplot, pdf.(dist[j], xplot), color=:red, linewidth=2)
        catch
            lines!(ax[j], xplot, updf.(dist[j], xplot) ./ normconst(dist[j], quadpts[1], quadpts[2]), color=:red, linewidth=2)
        end
        # axislegend(ax[j])
    end

    if thresh == :none 
        ttl *= " (top $nsamp samples)"
    else
        th = round(thresh, digits=1)
        # ttl *= " (threshold = $th)"
    end
    Label(fig[0,:], text=ttl, fontsize=20) # title
    return fig
end


function compute_invcdf(w::Vector{Float64}; thresh::Union{Nothing,Vector{Float64}}=nothing, rev=false)
    
    if thresh == nothing
        thresh = exp.(LinRange(log.(minimum(w)), log.(maximum(w)), 10))
    end

    P = Vector{Float64}(undef, length(thresh))
    for (i,t) in zip(1:length(thresh), thresh)
        if rev == false
            P[i] = length(findall(x -> x > t, w)) / length(w)
        else
            P[i] = length(findall(x -> x <= t, w)) / length(w)
        end
    end

    return P
end


# function get_medlimit(met_type::String, BD::String, nsamptot::Int64, repl::Int64; β=nothing)
#     M = length(nsamp_arr)
#     wmax = Vector{Float64}(undef, M)
#     wmin = Vector{Float64}(undef, M)

#     if β == nothing
#         metrics = JLD.load("data1/repl$nrepl/DW1D_$(BD)_nsamp=$nsamptot.jld")["metrics"]
#     else
#         metrics = JLD.load("data1/repl$nrepl/DW1D_$(BD)_nsamp=$nsamptot.jld")["metrics"][β]
#     end
#         wmax[j] = maximum(metrics[met_type])
#         wmin[j] = minimum(metrics[met_type])
#     end
#     return median(wmin), median(wmax)
# end


function get_medlimit(met_type::String, met_tup::Tuple)
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    
    wmax = [maximum(wi) for wi in w_arr]
    wmin = [minimum(wi) for wi in w_arr]
    
    return median(wmin), median(wmax)
end


# function get_maxlimit(met_type::String, BD::String, nsamp_arr::Vector{Int64}, nrepl_arr::Vector{Int64})
#     M = length(nsamp_arr)
#     wmax = Vector{Float64}(undef, M)
#     wmin = Vector{Float64}(undef, M)

#     for (j, nsamp, nrepl) in zip(1:M, nsamp_arr, nrepl_arr)
#         if β == nothing
#             metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
#         else
#             metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"][β]
#         end
#         wmax[j] = maximum(metrics[met_type][1])
#         wmin[j] = minimum(metrics[met_type][1])
#     end
#     return minimum(wmin), maximum(wmax)
# end


function get_maxlimit(met_type::String, met_tup::Tuple)
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    
    wmax = [maximum(wi) for wi in w_arr]
    wmin = [minimum(wi) for wi in w_arr]
    
    return minimum(wmin), maximum(wmax)
end


# function plot_IS_cdf(met_type::String, BD::String, nsamp_arr::Vector{Int64}, nrepl_arr::Vector{Int64}, ttl::String; β=nothing, limtype::Union{String, Tuple}=nothing)
#     M = length(nsamp_arr)
#     if limtype == nothing
#         wmin = 1e-8
#         _, wmax = get_maxlimit(met_type, BD, nsamp_arr, nrepl_arr; β=β)
#     elseif limtype == "med"; wmin, wmax = get_medlimit(met_type, BD, nsamp_arr, nrepl_arr; β=β);
#     elseif limtype == "max"; wmin, wmax = get_maxlimit(met_type, BD, nsamp_arr, nrepl_arr; β=β);
#     else; wmin = limtype[1]; wmax = limtype[2]; end

#     thresh = exp.(LinRange(log.(wmin), log.(wmax), 50))

#     fig = Figure(resolution = (450, 500))
#     ax = Axis(fig[1,1], xlabel="threshold (t)", ylabel="P[w > t]", xscale=log10, title=ttl)

    
#     for (k, nsamp, nrepl) in zip(1:M, nsamp_arr, nrepl_arr)
#         if β == nothing
#             metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
#         else
#             metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"][β]
#         end
#         w = metrics[met_type][1]
#         Pi = compute_invcdf(w; thresh=thresh)
#         # scatter!(ax, thresh, Pi, color=k, colormap=:Blues, colorrange=(0,M), label="n = $nsamp")
#         lines!(ax, thresh, Pi, color=k, colormap=:Blues, colorrange=(0,M), label="n = $nsamp")
#     end

#     axislegend(ax)
#     return fig
# end


function plot_IS_cdf(met_type::String, met_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String; limtype::Union{String, Vector}=nothing, rev=false, xticklab::Vector=nothing, logscl=true)
    colors = [:skyblue1, :seagreen, :goldenrod1, :darkorange2] 
    markers = [:hexagon, :circle, :diamond, :rect]

    M = length(met_tup)
    w_arr = [met_i[met_type] for met_i in met_tup]
    if limtype == nothing
        wmin = 1e-8
        _, wmax = get_maxlimit(met_type, met_tup)
    elseif limtype == "med"; wmin, wmax = get_medlimit(met_type, met_tup);
    elseif limtype == "max"; wmin, wmax = get_maxlimit(met_type, met_tup);
    else; wmin = limtype[1]; wmax = limtype[end]; end

    

    fig = Figure(resolution = (700, 700))
    if logscl == true
        thresh = exp.(LinRange(log.(wmin), log.(wmax), 50))
        threshscat = exp.(LinRange(log.(wmin), log.(wmax), 20))
        ax = Axis(fig[1,1], xlabel="threshold (t)", ylabel=ylab, xscale=log10, xticks=(limtype, xticklab), title=ttl)
    else
        thresh = Vector(LinRange(wmin, wmax, 50))
        threshscat = Vector(LinRange(wmin, wmax, 20))
        ax = Axis(fig[1,1], xlabel="threshold (t)", ylabel=ylab, xticks=(limtype, xticklab), title=ttl)
    end
    
    for (k, wi) in zip(1:M, w_arr)
        Pi = compute_invcdf(wi; thresh=thresh, rev=rev)
        Pis = compute_invcdf(wi; thresh=threshscat, rev=rev)
        lines!(ax, thresh, Pi, color=colors[k], linewidth=2)
        scatter!(ax, threshscat, Pis, color=colors[k], markersize=10, marker=markers[k], label=lab_tup[k])
    end

    if met_type == "wdiag"
        vlines!(ax, 5.0, linestyle=:dash, color=:red, label="threshold")
    end
    [vlines!(ax, t, color=(:black, 0.05)) for t in threshscat]

    axislegend(ax) # position=:lb)
    return fig
end


function import_data(dir::String, IntType::String, nsamp_arr::Vector{Int64}, nrepl::Int64; nsamptot::Int64=nsamp_arr[end])
    Cdict = Dict{Int64, Vector{Matrix{Float64}}}()
    
    for (k,nsamp) in zip(1:length(nsamp_arr), nsamp_arr)
        Cdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)
        for j = 1:nrepl
            Cj = JLD.load("$(dir)/repl$j/DW1D_$(IntType)_nsamp=$nsamptot.jld")["C"]
            Ckeys = collect(keys(Cj))
            Cdict[nsamp][j] = Cj[Ckeys[k]]
        end
    end

    return Cdict
end

function import_data(dir::String, IntType::String, nsamp_arr::Vector{Int64}, nrepl::Int64, β::Float64; nsamptot::Int64=nsamp_arr[end])
    Cdict = Dict{Int64, Vector{Matrix{Float64}}}()

    for (k,nsamp) in zip(1:length(nsamp_arr), nsamp_arr)
        Cdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)
        for j = 1:nrepl
            Cj = JLD.load("$(dir)/repl$j/DW1D_$(IntType)_nsamp=$nsamptot.jld")["C"][β]
            Ckeys = collect(keys(Cj))
            Cdict[nsamp][j] = Cj[Ckeys[k]]
        end
    end

    return Cdict
end

function compute_eigen_(Cdict::Dict, nsamp_arr::Vector{Int64}, nrepl::Int64, rdim::Int64)
    # initialize
    λdict = Dict{Int64, Vector{Vector{Float64}}}()
    Wdict = Dict{Int64, Vector{Matrix{Float64}}}()

    # compute eigenvalues and eigenvectors
    for nsamp in nsamp_arr
        λdict[nsamp] = Vector{Vector{Float64}}(undef, nrepl)
        Wdict[nsamp] = Vector{Matrix{Float64}}(undef, nrepl)

        for j = 1:nrepl
            _, λdict[nsamp][j], Wdict[nsamp][j] = select_eigendirections(Cdict[nsamp][j], rdim)
        end
    end

    return λdict, Wdict
end

function compute_val(Cref::Matrix{Float64}, Cdict::Dict, rdim::Int64)
    # extract values
    nsamp_arr = collect(keys(Cdict))
    nrepl = length(Cdict[nsamp_arr[1]])
    
    # compute eigenvalues and eigenvectors
    _, λref, _ = select_eigendirections(Cref, rdim)
    λdict, _ = compute_eigen_(Cdict, nsamp_arr, nrepl, rdim)

    # compute validation metrics
    val = Dict{String, Dict}()
    val["λ1_err"] = Dict{Int64, Vector{Float64}}()
    val["λ2_err"] = Dict{Int64, Vector{Float64}}()
    val["Forstner"] = Dict{Int64, Vector{Float64}}()
    val["WSD"] = Dict{Int64, Vector{Float64}}()

    for nsamp in nsamp_arr
        val["λ1_err"][nsamp] = [(λi[1] - λref[1]) / λref[1] for λi in λdict[nsamp]]
        val["λ2_err"][nsamp] = [(λi[2] - λref[2]) / λref[2] for λi in λdict[nsamp]]
        val["Forstner"][nsamp] = [ForstnerDistance(Cref, Ci) for Ci in Cdict[nsamp]]
        val["WSD"][nsamp] = [WeightedSubspaceDistance(Cref, Ci, rdim) for Ci in Cdict[nsamp]]
    end

    return val

end


function plot_sample_as(W::Matrix, rdim::Int64, πβ::Distribution, ny::Int64, r::Vector, B::Vector)
    W1, W2, π_y, π_z = compute_as(W, rdim, πβ)

    # draw samples
    ysamp = [rand(π_y) for i = 1:ny]
    θy = [W1*y + W2*π_z.μ for y in ysamp]
    θymat = reduce(hcat, θy)

    # compute energies
    energies = [Bi' * βi for Bi in B, βi in θy]

    # plot
    fig = Figure(resolution=(550, 500))
    ax = Axis(fig[1,1], xlabel="r (Å)", ylabel="E (eV)", title="Samples from active subspace (r=$rdim)")
    [lines!(ax, r, energies[:,j]) for j in 1:ny] # color=(:blue, 0.2)
    
    return fig

end


function plot_sample_as(W_tup::Tuple, rdim::Int64, πβ::Distribution, ny::Int64, B::Vector, lab_tup::Tuple)
    col = [:blue, :orange, :cyan, :red]
    N = length(W_tup)
    energies = Vector{Matrix{Float64}}(undef, N)

    for (i, W) in zip(1:N, W_tup)
        W1, W2, π_y, π_z = compute_as(W, rdim, πβ)

        # draw samples
        ysamp = [rand(π_y) for i = 1:ny]
        θy = [W1*y + W2*π_z.μ for y in ysamp]
        # θymat = reduce(hcat, θy)

        # compute energies
        energies[i] = [Bi' * βi for Bi in B, βi in θy]
    end

    # plot
    fig = Figure(resolution=(550, 500))
    ax = Axis(fig[1,1], xlabel="r (Å)", ylabel="E (eV)")
    for i = 1:N
        [lines!(ax, r, energies[i][:,j], color=(col[i], 0.2)) for j in 1:ny-1] 
        lines!(ax, r, energies[i][:,ny], color=(col[i], 0.2), label=lab_tup[i])
    end
    axislegend(ax)
    return fig

end


function plot_as_modes(W::Matrix, πβ::Distribution, ny::Int64, r::Vector, B::Vector)

    d = size(W, 1)
    if d % 2 != 0
        d = d + 1
        W = reduce(vcat, [W, zeros(d)])
    end

    fig = Figure(resolution=(900, 1300))
    ax = Vector{Axis}(undef, d)
    dims = Vector(1:d)

    for i = 1:d
        W1 = W[:,i:i] # type as matrix
        W2 = W[:, dims[Not(i)]]

         # sampling density of inactive variable z
        μz = W2' * πβ.μ
        Σz = Hermitian(W2' * πβ.Σ * W2) # + 1e-10 * I(βdim-rdim)
        π_z = MvNormal(μz, Σz)

        # compute sampling density
        μy = W1' * πβ.μ
        Σy = Hermitian(W1' * πβ.Σ * W1)
        π_y = MvNormal(μy, Σy)

        # draw samples
        ysamp = [rand(π_y) for i = 1:ny]
        θy = [W1*y + W2*π_z.μ for y in ysamp]

        # compute energies
        energies = [Bi' * βi for Bi in B, βi in θy]

        # define axis
        if i < d/2 + 1
            ax[i] = Axis(fig[i,1], xlabel="r (Å)", yticks=-0.006:0.002:0.00, limits = (1, 5, -0.008, 0.002), ylabel="E (eV)", title="mode $i")
        elseif i == Int(d/2)
            ax[i] = Axis(fig[i,1], xlabel="r (Å)", yticks=-0.006:0.002:0.00, limits = (1, 5, -0.008, 0.002), ylabel="E (eV)", title="mode $i")
        elseif i == d 
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="r (Å)", yticks=-0.006:0.002:0.00, limits = (1, 5, -0.008, 0.002), ylabel="E (eV)", title="mode $i")
        else
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="r (Å)", yticks=-0.006:0.002:0.00, limits = (1, 5, -0.008, 0.002), ylabel="E (eV)", title="mode $i")
        end
            
        # plot
        [lines!(ax[i], r, energies[:,j]) for j in 1:ny] # color=(:blue, 0.2)
    end
    fig
end


function plot_parameter_study(πβ::Distribution, n::Int64, r::Vector, B::Vector)

    d = size(πβ.μ, 1)
    W = I(d)
    if d % 2 != 0
        d = d + 1
        W = reduce(vcat, [W, zeros(d)])
    end

    fig = Figure(resolution=(1000, 1500))
    ax = Vector{Axis}(undef, d)
    dims = Vector(1:d)

    for i = 1:d
        W1 = W[:,i:i] # type as matrix
        W2 = W[:, dims[Not(i)]]

         # sampling density of inactive variable z
        μz = W2' * πβ.μ
        Σz = Hermitian(W2' * πβ.Σ * W2) # + 1e-10 * I(βdim-rdim)
        π_z = MvNormal(μz, Σz)

        # compute sampling density
        μy = W1' * πβ.μ
        Σy = 100*Hermitian(W1' * πβ.Σ * W1)
        π_y = MvNormal(μy, Σy)

        # draw samples
        ysamp = [rand(π_y) for i = 1:ny]
        θy = [W1*y + W2*π_z.μ for y in ysamp]

        # compute energies
        energies = [Bi' * βi for Bi in B, βi in θy]

        # define axis
        if i < d/2 + 1
            ax[i] = Axis(fig[i,1], ylabel="E (eV)", title="mode $i")
        elseif i == Int(d/2)
            ax[i] = Axis(fig[i,1], xlabel="r (Å)", ylabel="E (eV)", title="mode $i")
        elseif i == d 
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="r (Å)", ylabel="E (eV)", title="mode $i")
        else
            ax[i] = Axis(fig[i-Int(d/2),2], ylabel="E (eV)", title="mode $i")
        end
            
        # plot
        [lines!(ax[i], r, energies[:,j]) for j in 1:ny] # color=(:blue, 0.2)
    end
    fig
end
    
    # β = πβ.μ
    # σ = diag(πβ.Σ)
    # nβ = length(β)

    # fig = Figure(resolution=(550, 500))
    # ax = Axis(fig[1,1], xlabel="r (Å)", ylabel="E (eV)")
    
    # for i = 1:nβ
    #     perturb = Vector(LinRange(β[i] - 3*σ[i], β[i] + 3*σ[i], n))
    #     for p in perturb
    #         βi = deepcopy(β)
    #         βi[i] = p
    #         energies_i = [Bj' * βi for Bj in B]


function MCMC_traceplot(postsamp)

    d = size(postsamp, 1)
    if d % 2 != 0
        d = d + 1
        postsamp = reduce(vcat, [postsamp, zeros(size(postsamp, 2))'])
    end


    fig = Figure(resolution=(1800, 1000))
    ax = Vector{Axis}(undef, d)

    for i = 1:d
        if i < d/2 + 1
            ax[i] = Axis(fig[i,1], ylabel="z_$i")
        elseif i == Int(d/2)
            ax[i] = Axis(fig[i,1], xlabel="iteration (k)", ylabel="z_$i")
        elseif i == d 
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="iteration (k)", ylabel="z_$i")
        else
            ax[i] = Axis(fig[i-Int(d/2),2], ylabel="z_$i")
        end
        postsamp_i = postsamp[i,:]
        lines!(ax[i], 1:length(postsamp_i), postsamp_i)
    end
    fig
end



function MCMC_autocorrplot(postsamp; lags=0:1000)
    fig = Figure(resolution=(1000, 750))
    ax = Axis(fig[1,1], xlabel="lag (τ)", ylabel="autocorr(z_i)")
    for i = 1:d
        autocorr_i = autocor(postsamp[i,:], lags)
        lines!(ax, 1:length(autocorr_i), autocorr_i, label="z_$i")
    end
    axislegend(ax)
    fig
end


function plot_eigenvectors(W::Matrix)
    βdim = size(W, 1)
    fig = Figure(resolution=(600,1000))
    ax = Axis(fig[1,1], xlabel="dim.", ylabel="eigenvector i", xticks = 1:βdim, yticks=(2.5:2.5:2.5*βdim, string.(βdim:-1:1)), title="Eigenvectors")
    for i = 1:βdim
        scatterlines!(ax, 1:βdim, W[:,i] .+ 2.5*(βdim+1-i), color=RGB(0, i/βdim, 0), label="ϕ_$i")
    end
    # axislegend(ax, position=:lt)
    fig
end


custom_theme = Theme(
    font = "/Users/joannazou/Documents/MIT/Presentations/UNCECOMP 2023/InterFont/static/Inter-Regular.ttf",
    fontsize = 20,
    Axis = (
        xgridcolor = :white,
        ygridcolor = :white,
    )
)

