using CairoMakie, Colors, InvertedIndices


# set plot theme ##########################################################################################

custom_theme = Theme(
    font = "/Users/joannazou/Documents/MIT/Presentations/UNCECOMP 2023/InterFont/static/Inter-Regular.ttf",
    fontsize = 20,
    Axis = (
        xgridcolor = :white,
        ygridcolor = :white,
    )
)


# plot pdfs ##########################################################################################

function plot_pdf_with_sample_hist(d::Distribution, xplot::Vector, xsamp::Vector)
    with_theme(custom_theme) do
        fig = Figure(resolution = (700, 600))
        ax = Axis(fig[1, 1], xlabel="x", ylabel="pdf(x)", title="Comparison of distribution and samples")
        if hasapproxnormconst(d) == true
            lines!(ax, xplot, pdf.((d,), xplot, (GQint,)), label="true dist.")
        else
            lines!(ax, xplot, pdf.((d,), xplot), label="true dist.")
        end
        hist!(ax, xsamp, color=(:blue, 0.2), normalization=:pdf, bins=50, label="samples")

        axislegend(ax)
        return fig
    end
end


function plot_gibbs_pdf(θsamp::Vector{Vector{T}}, π0::Gibbs, xplot::Vector{T}, normint::Integrator; col::Union{Vector,Nothing} = nothing, ttl::String="") where T <: Real
    with_theme(custom_theme) do
        fig = Figure(resolution = (600, 600))
        ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)", title=ttl)
        for (i, θi) in enumerate(θsamp)
            πg = Gibbs(π0, β=1.0, θ=θi)
            if col != nothing
                lines!(ax, xplot, updf.((πg,), xplot) ./ normconst(πg, normint), color=col[i])
            else
                lines!(ax, xplot, updf.((πg,), xplot) ./ normconst(πg, normint))
            end
        end
        return fig
    end
end


function plot_integrand(θsamp::Vector{Vector{T}}, ϕ::Function, π0::Gibbs, xplot::Vector{T}, normint::Integrator, col::Vector; ttl::String="") where T <: Real
    with_theme(custom_theme) do
        fig = Figure(resolution = (600, 600))
        ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)", title=ttl)
        for (i, θi) in enumerate(θsamp)
            πg = Gibbs(π0, β=1.0, θ=θi)
            lines!(ax, xplot, ϕ.(xplot, (θi,)) .* updf.((πg,), xplot) ./ normconst(πg, normint), color=col[i])
            # lines!(ax, xplot, integrand(xplot, θi, ϕ, π0, GQint), color=col[i])
        end
        return fig
    end
end


function plot_multiple_pdfs(pdfs::Vector{<:Distribution}, xplot::Vector{T}, lbls::Vector{String}, cols::Vector; normint::Union{Integrator,Nothing}=nothing, ttl::String="") where T <: Real
    with_theme(custom_theme) do
        fig = Figure(resolution = (600, 600))
        ax = Axis(fig[1, 1],  xlabel="x", ylabel="pdf(x)", yticks=0:0.2:0.8, title=ttl)
        for (i, πi) in enumerate(pdfs)
            if hasupdf(πi)
                lines!(ax, xplot, updf.((πi,), xplot) ./ normconst(πi, normint), linewidth=2, color=cols[i], label=lbls[i]) 
            elseif hasapproxnormconst(πi)
                lines!(ax, xplot, pdf(πi, xplot, normint), linewidth=2, color=cols[i], label=lbls[i]) 
            else
                lines!(ax, xplot, pdf.((πi,), xplot), linewidth=2, color=cols[i], label=lbls[i]) 
            end
        end
        axislegend(ax)
        return fig
    end
end

# plot parameter space (2D) ##########################################################################################

function plot_parameter_space(θ1rng::Vector, θ2rng::Vector, Q::Matrix, P::Matrix, θsamp::Matrix, θysamp::Matrix)
    with_theme(custom_theme) do
        fig = Figure(resolution = (700, 600))
        ax1 = Axis(fig[1, 1][1, 1], xlabel="θ₁", ylabel="θ₂", title="Parameter space")
        # QoI distribution
        hm = heatmap!(ax1, θ1rng, θ2rng, Q, colorrange=(-20, 1)) 
        # contours of ρ(θ)
        contour!(ax1, θ1rng, θ2rng, P, color=(:white, 0.5), linewidth=2) #  levels=0:0.1:0.8) # exp.(LinRange(log(1e-3), log(1), 8))
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


function plot_parameter_space(θ1rng::Vector, θ2rng::Vector, Q::Matrix, P::Matrix, θsamp::Vector, θysamp::Vector)
    θmat = reduce(hcat, θsamp)
    θymat = reduce(hcat, θysamp)
    return plot_parameter_space(θ1rng, θ2rng, Q, P, θmat, θymat)
end


function plot_parameter_space(θ1rng::Vector, θ2rng::Vector, Q::Matrix, P::Matrix, θsamp::Matrix, as_arr::Vector{Subspace}, col_arr::Vector{Symbol}, lab_arr::Vector{String}; nas=200)
    with_theme(custom_theme) do
        fig = Figure(resolution = (900, 600))
        ax1 = Axis(fig[1, 1][1, 1], xlabel="θ₁", ylabel="θ₂", title="Parameter space")
        # QoI distribution
        hm = heatmap!(ax1, θ1rng, θ2rng, Q) # colorrange=(-300, 1)
        # contours of ρ(θ)
        contour!(ax1, θ1rng, θ2rng, P, color=(:white, 0.5), linewidth=2) #  levels=0:0.1:0.8) # exp.(LinRange(log(1e-3), log(1), 8))
        # random samples from ρ(θ)
        scatter!(ax1, θsamp[1,:], θsamp[2,:], color=(:white, 0.75), markersize=7, label="original density")
        # random samples from AS
        for (j, as) in enumerate(as_arr)
            _, θy = sample_as(nas, as)
            θy = reduce(hcat, θy)
            scatter!(ax1, θy[1,:], θy[2,:], color=(col_arr[j], 0.75), markersize=7, label="AS "*lab_arr[j])
        end
        # color scale
        Colorbar(fig[1, 1][1, 2], hm, label="Q(θ)")
        fig[1, 2] = Legend(fig, ax1, framevisible = false)
        # axislegend(ax1, position=:lt)
        return fig
    end
end

# plot validation metric ##########################################################################################

function plot_val_metric(val_type::String, val_tup::Union{Tuple,Vector}, lab_tup::Union{Tuple,Vector}, ylab::String, ttl::String; logscl=true)
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
            title=ttl, xscale=log2) #  yticks=0.0:0.005:0.025) 
        end
        offset_ratio = [-0.08, -0.04, 0, 0.04, 0.08, 0.12]
        # offsets = Point[(-20, 0), (-10, 0), (0,0), (10,0), (20,0)]
        # offsets = [exp.(os) for os in offsets]
        for (i, val_i, lab_i) in zip(1:length(val_tup), val_tup, lab_tup)
            val_med = [median(val_i[val_type][k]) for k in nsamp_arr]
            val_hi = [maximum(val_i[val_type][k]) for k in nsamp_arr] # [percentile(val_i[val_type][k], 75) for k in nsamp_arr] # .+ 1e-6
            val_lo = [minimum(val_i[val_type][k]) for k in nsamp_arr]# [percentile(val_i[val_type][k], 25) for k in nsamp_arr] # .+ 1e-6
            offsets = offset_ratio[i]*nsamp_arr

            for k = 1:N
                nsamp = nsamp_arr[k]
                if k == 1
                    scatter!(ax, [nsamp + offsets[k]], [val_med[k]], color=colors[i], marker=markers[i], markersize=15, label=lab_i)
                else
                    scatter!(ax, [nsamp + offsets[k]], [val_med[k]], color=colors[i], marker=markers[i], markersize=15)
                end
                rangebars!(ax, [nsamp + offsets[k]], [val_lo[k]], [val_hi[k]], color=colors[i], linewidth=2) #  whiskerwidth=nsamp / 2^(4*k))
            end
        end
        fig[1, 2] = Legend(fig, ax, framevisible = false)
        # axislegend(ax, position=:ct)
        return fig
    end
end


# plot IS diagnostic measures ##########################################################################################

function plot_IS_diagnostic(met_type::String, met_tup::Union{Tuple,Vector}, lab_tup::Union{Tuple,Vector}, ylab::String, ttl::String; logscl=false)
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


function plot_IS_diagnostic(met_type::String, met_tup::Union{Tuple,Vector}, lab_tup::Union{Tuple,Vector}, ylab::String, ttl::String, W::Matrix; logscl=false)
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


function plot_IS_samples(πg0::Gibbs, met_type::String, met_tup::Union{Tuple,Vector}, lab_tup::Union{Tuple,Vector}, dist::Union{Tuple,Vector}, ttl::String, xlim::Vector, quadpts::Union{Tuple,Vector}; rev=false, nsamp=100, thresh=:none)
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


function get_medlimit(met_type::String, met_tup::Union{Tuple,Vector})
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    
    wmax = [maximum(wi) for wi in w_arr]
    wmin = [minimum(wi) for wi in w_arr]
    
    return median(wmin), median(wmax)
end


function get_maxlimit(met_type::String, met_tup::Union{Tuple,Vector})
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    
    wmax = [maximum(wi) for wi in w_arr]
    wmin = [minimum(wi) for wi in w_arr]
    
    return minimum(wmin), maximum(wmax)
end


function plot_IS_cdf(met_type::String, met_tup::Union{Tuple,Vector}, lab_tup::Union{Tuple,Vector}, ylab::String, ttl::String; limtype::Union{String, Vector}=nothing, rev=false, xticklab::Vector=nothing, logscl=true)
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


# plot eigenspectrum ##########################################################################################

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


function plot_eigenspectrum(Cref::Matrix, C_tup::Union{Tuple,Vector}, lab_tup::Union{Tuple,Vector}; logscl=true)
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


function plot_eigenspectrum(asr::T,
    asis_arr::Vector{Vector{T}},
    labs::Vector{String},
    colors::Vector{Symbol};
    markers::Vector{Symbol}=[:circle for i = 1:length(metvec)],
    ls=[nothing for i = 1:length(asis_arr)]) where T <: Subspace

    d = length(asr.λ)

    with_theme(custom_theme) do
        fig = Figure(resolution=(900,600))
        ax = Axis(fig[1,1],
        title="Spectrum of matrix C",
        xlabel="index i",
        ylabel="eigenvalue (λi)",
        xticks = 1:d,
        yscale=log10
        )
        # plot reference
        sc = Vector(undef, length(asis_arr)+1)
        sc[1] = scatterlines!(ax, 1:d, asr.λ, color=:black, label="Ref.")

        # plot comparison
        for (i,asis) in enumerate(asis_arr)
            λ_arr = [as.λ for as in asis]
            λmean = mean(λ_arr)
            λmin = minimum(λ_arr)
            λmax = maximum(λ_arr)

            sc[i+1] = scatterlines!(ax, 1:d, λmean, color=colors[i], linestyle=ls[i], marker=markers[i], label=labs[i]) # linewidth=(7-i)*0.8,
            band!(ax, 1:d, λmin, λmax, color=(colors[i], 0.3))
        end
        # axislegend(ax)
        Legend(fig[1,2], sc, reduce(vcat, ["Ref.", labs]))

        return fig
    end
end


# plot cosine similarity ##########################################################################################

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

function plot_cossim(Cref::Matrix{<:Real}, Ctup::Union{Tuple,Vector}, labs::Union{Tuple,Vector})
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


function plot_cossim(asr::T,
    asis_arr::Vector{Vector{T}},
    labs::Vector{String},
    colors::Vector{Symbol};
    markers::Vector{Symbol}=[:circle for i = 1:length(metvec)],
    ls=[nothing for i = 1:length(asis_arr)]) where T <: Subspace

    d = size(asr.C, 1)

    with_theme(custom_theme) do
        fig = Figure(resolution=(600,600))
        ax = Axis(fig[1,1],
        title="Cosine similarity",
        xlabel="eigenvector (ϕi)",
        ylabel="cos. sim.",
        xticks = 1:d
        )

        for (i,asis) in enumerate(asis_arr)
            W_arr = [as.W1 for as in asis]
            cs_arr = [[abs.(asr.W1[:,r]'*as.W1[:,r]) for r = 1:d] for as in asis]
            csmean = mean(cs_arr)
            csmin = minimum(cs_arr)
            csmax = maximum(cs_arr)

            scatterlines!(ax, 1:d, csmean, color=colors[i], label=labs[i], marker=markers[i], linestyle=ls[i])
            # band!(ax, 1:d, csmin, csmax, color=(colors[i], 0.3))

        end
        # axislegend(ax)
        return fig
    end
end


function plot_cossim_mat(asr::T,
    asis_arr::Vector{Vector{T}},
    labs::Vector{String}
    ) where T <: Subspace

    d = size(asr.C, 1)

    with_theme(custom_theme) do
        fig = Figure(resolution=(1200,1000))

        ax1 = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        ax2 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]

        ax = Vector{Axis}(undef, length(asis_arr))
        for (i,asis) in enumerate(asis_arr)

            ax[i] = Axis(fig[ax1[i],ax2[i]],
                    title=labs[i],
                    xticklabelsvisible=false,
                    yticklabelsvisible=false
                    )
            as = asis[1]
            cs_mat = [(abs.(asr.W1[:,i]'*as.W1[:,j])) for i=1:d, j=1:d]
            hm = heatmap!(ax[i], 1:d, 1:d, cs_mat[:, end:-1:1], colorrange=(0,1))
            if i == length(asis_arr); Colorbar(fig[:,5], hm); end
        end

        return fig
    end
end

# plot eigenmodes ##########################################################################################

function plot_as_modes(W::Matrix,
    ρθ::Distribution,
    ny::Int64,
    π0::Gibbs,
    xplot::Vector,
    normint::Integrator)

    d = size(W, 1)

    with_theme(custom_theme) do
        fig = Figure(resolution=(900, 1300))
        ax = Vector{Axis}(undef, d)
        dims = Vector(1:d)

        for i = 1:d
            W1 = W[:,i:i] # type as matrix
            W2 = W[:, dims[Not(i)]]

            # compute sampling densities
            π_y = compute_marginal(W1, ρθ)
            π_z = compute_marginal(W2, ρθ)

            # draw samples
            ysamp = [rand(π_y) for i = 1:ny]
            θy = [W1*y + W2*π_z.μ for y in ysamp]

            # define axis
            ax[i] = Axis(fig[i,1], xlabel="x", ylabel="pdf(x)", title="mode $i")

            # plot
            for (j, θj) in enumerate(θy)
                πg = Gibbs(π0, β=1.0, θ=θj)
                lines!(ax[i], xplot, updf.((πg,), xplot) ./ normconst(πg, normint))
            end
        end
        return fig
    end
end


function plot_as_modes(as::Subspace,
    ρθ::Distribution,
    ny::Int64,
    r::Vector, # interatomic distances
    B::Vector; # descriptors
    scl=1) 

    # compute subspace
    d = size(as.C, 1)
    _, _, W = compute_eigenbasis(as.C)

    return plot_as_modes(W, ρθ, ny, r, B; scl=scl)
end


function plot_parameter_study(ρθ::Distribution,
           ny::Int64,
           r::Vector, # interatomic distances
           B::Vector; # descriptors
           scl=1)
    d = length(ρθ.μ)
    W = Matrix(I(d))
    return plot_as_modes(W, ρθ, ny, r, B, scl=scl)
end


# plot error histograms ######################################################################

function plot_error_scatter(vref::Vector{T},
    v_arr::Vector{Vector{T}},
    θsamp::Vector{T},
    colors::Vector{Symbol}, 
    labs::Vector{String},
    ylab::String,
    ttl::String;
    logscl=false) where T <: Vector{<:Real}

    err = [log.(abs.(EuclideanDistance.(vref, v_i))) for v_i in v_arr]
    nv = length(err)

    cmin = minimum([minimum(e) for e in err])       # minimum value for colorbar
    cmax = maximum([maximum(e) for e in err])       # maximum value for colorbar

    if logscl == true
        cmin = log.(cmin)
        cmax = log.(cmax)
        logerr = [log.(e) for e in err]
    end

    θmat = reduce(hcat, θsamp)

    with_theme(custom_theme) do
    # initialize figure
    fig = Figure(resolution=(450*nv, 500)) # Figure(resolution=(1600, 1300))
    ax = Vector{Axis}(undef, nv)

    for i = 1:nv
        ei = err[i]
        
        ax[i] = Axis(fig[1,i][1,1], xlabel="θ₁", ylabel="θ₂",
        title=labs[i]) 
        # plot 2D scatterplot and colorbar
        sc = scatter!(ax[i], θmat[1,:], θmat[2,:], markersize=7, color=ei, colormap=Reverse(:Blues), colorrange=(cmin,cmax)) 
        if i == nv
            Colorbar(fig[1,i][1,2], sc, label=ylab)
        end  
    end
    Label(fig[0,:], text=ttl, fontsize=20)
    return fig
    end
end


function plot_error_hist(err_pred::Vector{Float64})
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="error [eV]", ylabel="count")
    hist!(ax, err_pred, bins=30)
    return fig
end

function plot_error_hist(vref::Vector{T},
                    v_arr::Vector{Vector{T}},
                    colors::Vector{Symbol}, 
                    labs::Vector{String},
                    ylab::String,
                    ttl::String;
                    logscl=false) where T <: Vector{<:Real}

    err = [log.(abs.(EuclideanDistance.(vref, v_i))) for v_i in v_arr]

    fig = plot_staggered_hist(err, labs, colors, ylab, ttl, logscl=logscl)
    return fig
end

function plot_staggered_hist(vecs::Vector{T},
    xticklabs::Vector,
    colors::Vector{Symbol},
    ylab::String,
    ttl::String;
    logscl=false) where T <: Vector{<:Real}

    xoff = LinRange(0, 1, length(vecs))
    xscl = - 0.8 * 1/length(vecs) 

    fig = Figure(resolution=(800,400))

    if logscl == false
        ax = Axis(fig[1, 1], xticks=(xoff, xticklabs), ylabel=ylab, title=ttl)
    else
        ax = Axis(fig[1, 1], xticks=(xoff, xticklabs), ylabel=ylab, title=ttl, yscale=log10)
        vecs = [abs.(v) .+ 1e-10 for v in vecs]
    end
    ylims!(ax, -15, 10)

    [hist!(ax, v,
            scale_to=xscl,
            color=(colors[i], 0.7),
            # strokecolor=colors[i],
            # strokewidth=2,
            # strokearound=true,
            offset=xoff[i],
            bins=100,
            direction=:x) for (i,v) in enumerate(vecs)]

    return fig
end


# plot MCMC diagnostics ####################################################################################################

function plot_mcmc_trace(postsamp::Matrix)

    d = size(postsamp, 1)
    if d % 2 != 0
        d = d + 1
        postsamp = reduce(vcat, [postsamp, zeros(size(postsamp, 2))'])
    end


    fig = Figure(resolution=(1800, 1000))
    ax = Vector{Axis}(undef, d)

    for i = 1:d
        if i < d/2 + 1
            ax[i] = Axis(fig[i,1], ylabel="state $i")
        elseif i == Int(d/2)
            ax[i] = Axis(fig[i,1], xlabel="step (k)", ylabel="state $i")
        elseif i == d 
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="step (k)", ylabel="state $i")
        else
            ax[i] = Axis(fig[i-Int(d/2),2], ylabel=" state $i")
        end
        postsamp_i = postsamp[i,:]
        lines!(ax[i], 1:length(postsamp_i), postsamp_i)
    end
    fig
end


function plot_mcmc_trace(postsamp::Vector{Vector})
    postsamp = reduce(hcat, postsamp)
    return plot_mcmc_trace(postsamp)
end


function plot_mcmc_autocorr(postsamp::Matrix; lags=0:1000)
    d = size(postsamp, 1)
    fig = Figure(resolution=(1000, 750))
    ax = Axis(fig[1,1], xlabel="lag (τ)", ylabel="autocorr.")
    for i = 1:d
        autocorr_i = autocor(postsamp[i,:], lags)
        lines!(ax, 1:length(autocorr_i), autocorr_i, label="state $i")
    end
    axislegend(ax)
    fig
end


function plot_mcmc_autocorr(postsamp::Vector{Vector{Float64}})
    postsamp = reduce(hcat, postsamp)
    return plot_mcmc_autocorr(postsamp)
end


function plot_mcmc_marginals(postsamp::Matrix; plottitle="")
    nθ = size(postsamp, 1)

    fig = Figure(resolution = (1000, 1000))
    # Label(fig[-1,1:nθ], text = plottitle, textsize = 20)
    ax = Matrix{Axis}(undef, (nθ, nθ))

    for j = 1:nθ
        for i = j:nθ
            # set axes
            # if i == nθ
            #     ax[i,j] = Axis(fig[i+1,j][1,1],
            #                     xticksvisible=false,
            #                     xticklabelsvisible=false,
            #                     yticksvisible=false,
            #                     yticklabelsvisible=false)
            if j == i
                ax[i,j] = Axis(fig[i+1,j][1,1], title="x$i",
                                xticksvisible=false,
                                xticklabelsvisible=false,
                                yticksvisible=false,
                                yticklabelsvisible=false)
            else
                ax[i,j] = Axis(fig[i+1,j][1,1],
                                xticksvisible=false,
                                xticklabelsvisible=false,
                                yticksvisible=false,
                                yticklabelsvisible=false)
            end

            if i != j
                scatter!(ax[i,j], postsamp[i,:], postsamp[j,:], color=(:blue, 0.05))

            else # i == j
                hist!(ax[i,i], postsamp[i,:], bins=50, color=(:blue, 0.5))
            end
        end
    end
    fig

end


function plot_mcmc_marginals(postsamp::Vector{Vector{Float64}}; plottitle="")
    postsamp = reduce(hcat, postsamp)
    return plot_mcmc_marginals(postsamp)
end
