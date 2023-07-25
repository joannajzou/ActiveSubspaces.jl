using CairoMakie, Colors, InvertedIndices


custom_theme = Theme(
    font = "/Users/joannazou/Documents/MIT/Presentations/UNCECOMP 2023/InterFont/static/Inter-Regular.ttf",
    fontsize = 20,
    Axis = (
        xgridcolor = :white,
        ygridcolor = :white,
    )
)

function plot_pdf_with_sample_hist(d::Distribution, xplot::Vector{T}, xsamp::Vector{T}; ξx=:none, wx=:none) where T <: Real
    with_theme(custom_theme) do
        fig = Figure(resolution = (700, 600))
        ax = Axis(fig[1, 1], xlabel="x", ylabel="pdf(x)", title="Comparison of distribution and samples")
        if hasupdf(d) == true
            lines!(ax, xplot, updf.((d,), xplot) ./ normconst(d, ξx, wx), label="true dist.")
            hist!(ax, xsamp, color=(:blue, 0.2), normalization=:pdf, bins=50, label="samples")
        else
            lines!(ax, xplot, pdf.(d, xplot), label="true dist.")
            hist!(ax, xsamp, color=(:blue, 0.2), normalization=:pdf, bins=50, label="samples")
        end
        
        return fig
    end
end

function plot_parameter_space(θ1rng::Vector, θ2rng::Vector, Q::Matrix, P::Matrix, θsamp::Matrix, θysamp::Matrix)
    with_theme(custom_theme) do
        fig = Figure(resolution = (700, 600))
        ax1 = Axis(fig[1, 1][1, 1], xlabel="θ₁", ylabel="θ₂", title="Parameter space")
        # QoI distribution
        hm = heatmap!(ax1, θ1rng, θ2rng, Q) 
        # contours of ρ(θ)
        contour!(ax1, θ1rng, θ2rng, P, color=(:white, 0.25), linewidth=2, levels=0:0.1:0.8) # exp.(LinRange(log(1e-3), log(1), 8))
        # random samples from ρ(θ)
        scatter!(ax1, θsamp[1,:], θsamp[2,:], color=(:white, 0.75), markersize=7, label="original density")
        # random samples from AS
        scatter!(ax1, θysamp[1,:], θysamp[2,:], color=(:red, 0.75), markersize=7, label="active subspace")
        # color scale
        Colorbar(fig[1, 1][1, 2], hm, label="Q(θ)")
        # axislegend(ax1, position=:lt)
        return fig
    end
end



function plot_gibbs_pdf(θsamp::Vector{Vector{T}}, π0::Gibbs, xplot::Vector{T}, ξx::Vector{T}, wx::Vector{T}, col::Vector; ttl::String="") where T <: Real
    with_theme(custom_theme) do
        fig = Figure(resolution = (600, 600))
        ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)", title=ttl)
        for (i, θi) in enumerate(θsamp)
            πg = Gibbs(π0, β=1.0, θ=θi)
            if modnum == 1; lines!(ax, xplot, updf.((πg,), xplot) ./ normconst(πg, ξx, wx), color=col[i])
            else; lines!(ax, xplot, updf.((πg,), xplot) ./ normconst(πg, ξx, wx), color=col[i]); end
        end
        return fig
    end
end

function plot_gibbs_pdf(θsamp::Vector{Vector{T}}, π0::Gibbs, xplot::Vector{T}, ξx::Vector{T}, wx::Vector{T}; ttl::String="") where T <: Real
    with_theme(custom_theme) do
        fig = Figure(resolution = (600, 600))
        ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)", title=ttl)
        for (i, θi) in enumerate(θsamp)
            πg = Gibbs(π0, β=1.0, θ=θi)
            if modnum == 1; lines!(ax, xplot, updf.((πg,), xplot) ./ normconst(πg, ξx, wx))
            else; lines!(ax, xplot, updf.((πg,), xplot) ./ normconst(πg, ξx, wx)); end
        end
        return fig
    end
end

function plot_multiple_pdfs(pdfs::Vector{<:Distribution}, xplot::Vector{T}, ξx::Vector{T}, wx::Vector{T}, lbls::Vector{String}, cols::Vector; ttl::String="") where T <: Real
    with_theme(custom_theme) do
        fig = Figure(resolution = (600, 600))
        ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)", yticks=0:0.2:0.8, title=ttl)
        for (i, πi) in enumerate(pdfs)
            if hasupdf(πi)
                lines!(ax, xplot, updf.((πi,), xplot) ./ normconst(πi, ξx, wx), linewidth=2, color=cols[i], label=lbls[i]) 
            else
                lines!(ax, xplot, pdf.(πi, xplot), linewidth=2, color=cols[i], label=lbls[i]) 
            end
        end
        axislegend(ax)
        return fig
    end
end



function plot_val_metric(val_type::String, val_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String; logscl=true)
    colors = [:mediumpurple4, :skyblue1, :seagreen, :goldenrod1, :darkorange2, :firebrick2] 
    markers = [:utriangle, :hexagon, :circle, :diamond, :rect, :star]
    nsamp_arr = collect(keys(val_tup[1][val_type]))
    N = length(nsamp_arr)

    with_theme(custom_theme) do
        fig = Figure(resolution=(900, 600))
        if logscl == true
            ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel=ylab, xticks=(nsamp_arr, [string(nsamp) for nsamp in nsamp_arr]),
            title=ttl, xscale=log10, yscale=log10)   
        else
            ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel=ylab, xticks=(nsamp_arr, [string(nsamp) for nsamp in nsamp_arr]),
            title=ttl, xscale=log10) #  yticks=0.0:0.005:0.025) 
        end
        offset_ratio = [-0.08, -0.04, 0, 0.04, 0.08, 0.12]
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
        return fig
    end
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

    with_theme(custom_theme) do
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
end


function plot_IS_diagnostic(met_type::String, met_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String, W::Matrix; logscl=false)
    w_arr = [met_i[met_type] for met_i in met_tup]
    nIS = length(w_arr)

    cmin = minimum([minimum(wi) for wi in w_arr])       # minimum value for colorbar
    cmax = maximum([maximum(wi) for wi in w_arr])       # maximum value for colorbar

    θsamp = met_tup[1]["θsamp"]
    θmat = reduce(hcat, θsamp)
    ymat = W' * θmat

    if logscl == true
        cmin = log.(cmin)
        cmax = log.(cmax)
        w_arr = [log.(wi) for wi in w_arr]
    end

    with_theme(custom_theme) do
        # initialize figure
        fig = Figure(resolution=(450*nIS, 500)) # Figure(resolution=(1600, 1300))
        ax = Vector{Axis}(undef, nIS)
        
        for i = 1:nIS
            wi = w_arr[i]
            
            ax[i] = Axis(fig[1,i][1,1], xlabel="y₁", ylabel="y₂",
            title=lab_tup[i]) 
            # plot 2D scatterplot and colorbar
            sc = scatter!(ax[i], ymat[1,:], ymat[2,:], markersize=7, color=wi, colormap=Reverse(:Blues), colorrange=(cmin,cmax)) 
            if i == nIS
                Colorbar(fig[1,i][1,2], sc, label=ylab)
            end  
        end
        Label(fig[0,:], text=ttl, fontsize=20)
        return fig
    end
end


function plot_IS_samples(πg0::Gibbs, met_type::String, met_tup::Tuple, lab_tup::Tuple, dist::Tuple, ttl::String, xlim::Vector, quadpts::Tuple; rev=false, nsamp=100, thresh=:none)
    w_arr = [met_i[met_type] for met_i in met_tup]
    θsamp = met_tup[1]["θsamp"]
    nIS = length(w_arr)

    with_theme(custom_theme) do
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

            ax[j] = Axis(fig[1,j],  xlabel="x", ylabel="pdf(x)", limits=(xlim[1], xlim[2], -0.02, 2), title=lab_tup[j])
            xplot = LinRange(xlim[1], xlim[2], 1000)

            # plot first nsamp samples 
            if m != 0
                for (i, θi) in enumerate(θsamp[ids])
                    πg = Gibbs(πg0, β=1.0, θ=θi)
                    if i == 1
                        lines!(ax[j], xplot, updf.((πg,), xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:steelblue, 0.1)) # label="prop: $(propsamp)%") 
                    else
                        lines!(ax[j], xplot, updf.((πg,), xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:steelblue, 0.1)) 
                    end
                end
            else
                lines!(ax[j], xplot, zeros(length(xplot)), color=(:steelblue, 0.001), label="prop: 0%") 
            end
            # plot biasing distribution
            try
            lines!(ax[j], xplot, pdf.((dist[j],), xplot), color=:red, linewidth=2)
            catch
                lines!(ax[j], xplot, updf.((dist[j],), xplot) ./ normconst(dist[j], quadpts[1], quadpts[2]), color=:red, linewidth=2)
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
end


function compute_invcdf(w::Vector{Float64}; thresh::Union{Nothing,Vector{Float64}}=nothing, rev=false)
    
    if thresh == nothing
        thresh = exp.(LinRange(log.(minimum(w)), log.(maximum(w)), 10))
    end

    P = Vector{Float64}(undef, length(thresh))
    for (i,t) in enumerate(thresh)
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
    colors = [:skyblue1, :seagreen, :goldenrod1, :darkorange2, :firebrick2] 
    markers = [:hexagon, :circle, :diamond, :rect, :star]

    M = length(met_tup)
    w_arr = [met_i[met_type] for met_i in met_tup]
    if limtype == nothing
        wmin = 1e-8
        _, wmax = get_maxlimit(met_type, met_tup)
    elseif limtype == "med"; wmin, wmax = get_medlimit(met_type, met_tup);
    elseif limtype == "max"; wmin, wmax = get_maxlimit(met_type, met_tup);
    else; wmin = limtype[1]; wmax = limtype[end]; end

    
    with_theme(custom_theme) do
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
        
        for (k, wi) in enumerate(w_arr)
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
end


function plot_eigenspectrum(λ::Vector{Float64})
    fig = Figure(resolution=(500,500))
    ax = Axis(fig[1,1],
            title="Spectrum of gradient covariance matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λi)",
            yscale=log10,
            xgridvisible=false,
            ygridvisible=false)
    scatterlines!(ax, 1:length(λ), λ)
    return fig
end


function plot_eigenspectrum(λ::Vector{T}; predlab=1:length(λ)) where T <: Vector{<:Real}
    colors = [:black, :mediumpurple4, :skyblue1, :seagreen, :goldenrod1, :darkorange2, :firebrick2] 
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Spectrum of matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λ_i)",
            yscale=log10)

    for k = 1:length(λ)
        scatterlines!(ax, 1:length(λ[k]), λ[k], color=colors[k], label=predlab[k])
    end
    axislegend(ax)
    return fig
end


function plot_eigenspectrum(Cref::Matrix, C_tup::Tuple, lab_tup::Tuple; logscl=true)
    colors = [:mediumpurple4, :skyblue1, :seagreen, :goldenrod1, :darkorange2, :firebrick2] 
    markers = [:utriangle, :hexagon, :circle, :diamond, :rect, :star]
    N = length(C_tup)

    # compute eigenvalues
    _, λref, _ = compute_eigenbasis(Cref)
    d = length(λref)

    with_theme(custom_theme) do
        fig = Figure(resolution=(900, 600))
        if logscl == true
            ax = Axis(fig[1,1], ylabel="eigenvalue (λi)", xticks=1:length(λref), yscale=log10)   
        else
            ax = Axis(fig[1,1], ylabel="eigenvalue (λi)", xticks=1:length(λref)) #  yticks=0.0:0.005:0.025) 
        end
        offsets = [-0.08, -0.04, 0, 0.04, 0.08, 0.12]

        scatter!(ax, 1:d, λref, color=:black, markersize=15, label="ref.")
        for (j,Cj) in enumerate(C_tup)
            λj = [compute_eigenbasis(C)[2] for C in Cj]

            λ_med = percentile(λj, 50)
            λ_hi = percentile(λj, 75)
            λ_lo = percentile(λj, 25)

            for k = 1:d
                if k == 1
                    scatter!(ax, [k + offsets[j]], [λ_med[k]], color=colors[j], marker=markers[j], markersize=15, label=lab_tup[j])
                else
                    scatter!(ax, [k + offsets[j]], [λ_med[k]], color=colors[j], marker=markers[j], markersize=15)
                end
                rangebars!(ax, [k + offsets[j]], [λ_lo[k]], [λ_hi[k]], color=colors[j], linewidth=2, whiskerwidth=0.5)
            end
        end
        fig[1, 2] = Legend(fig, ax, framevisible = false)
        # axislegend(ax, position=:ct)
        return fig
    end
end


function plot_cossim(Cref::Matrix{<:Real}, C::Matrix{<:Real})
    dim = size(Cref, 1)
    _, _, Wref = compute_eigenbasis(Cref)
    _, _, W = compute_eigenbasis(C)
    
    cossim = [Wref[:,i]'*W[:,i] for i = 1:dim]
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Cosine similarity",
            xlabel="eigenvector (ϕi)",
            xticks = 1:dim,
            ylabel="cos. sim.",
            # yscale=log10
            )

    scatterlines!(ax, 1:dim, cossim)
    return fig, cossim
end

function plot_cossim(Cref::Matrix{<:Real}, Ctup::Tuple, labs::Tuple)
    colors = [:mediumpurple4, :skyblue1, :seagreen, :goldenrod1, :darkorange2, :firebrick2] 
    dim = size(Cref, 1)
    _, _, Wref = compute_eigenbasis(Cref)
    
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Cosine similarity",
            xlabel="eigenvector (ϕi)",
            xticks = 1:dim,
            ylabel="cos. sim.",
            # yscale=log10
            )

    for (j,C) in enumerate(Ctup)
        _, _, W = compute_eigenbasis(C)
        cossim = [Wref[:,i]'*W[:,i] for i = 1:dim]
        scatterlines!(ax, 1:dim, cossim, color=colors[j], label=labs[j])
    end
    axislegend(ax)
    return fig
end



function plot_as_modes(W::Matrix, ρθ::Distribution, ny::Int64, π0::Gibbs, xplot::Vector, ξx::Vector, wx::Vector)

    d = size(W, 1)

    fig = Figure(resolution=(900, 1300))
    ax = Vector{Axis}(undef, d)
    dims = Vector(1:d)

    for i = 1:d
        W1 = W[:,i:i] # type as matrix
        W2 = W[:, dims[Not(i)]]

         # sampling density of inactive variable z
        μz = W2' * ρθ.μ
        Σz = Hermitian(W2' * ρθ.Σ * W2) # + 1e-10 * I(βdim-rdim)
        π_z = MvNormal(μz, Σz)

        # compute sampling density
        μy = W1' * ρθ.μ
        Σy = Hermitian(W1' * ρθ.Σ * W1)
        π_y = MvNormal(μy, Σy)

        # draw samples
        ysamp = [rand(π_y) for i = 1:ny]
        θy = [W1*y + W2*π_z.μ for y in ysamp]

        # define axis
        ax[i] = Axis(fig[i,1], xlabel="x", ylabel="pdf(x)", title="mode $i")
            
        # plot
        for (j, θj) in enumerate(θy)
            πg = Gibbs(π0, β=1.0, θ=θj)
            lines!(ax[i], xplot, updf.((πg,), xplot) ./ normconst(πg, ξx, wx))
        end
    end
    fig
end