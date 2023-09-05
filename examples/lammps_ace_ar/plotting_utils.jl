using CairoMakie
using Colors
using StatsBase
using InvertedIndices


custom_theme = Theme(
    font = "/Users/joannazou/Documents/MIT/Presentations/UNCECOMP 2023/InterFont/static/Inter-Regular.ttf",
    fontsize = 20,
    Axis = (
        xgridcolor = :white,
        ygridcolor = :white,
    )
)


function plot_energy(e_pred::Vector{Float64}, e_true)
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10

    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="E ref. [eV/atom]", ylabel="E pred. [eV/atom]")
    scatter!(ax, e_true, e_pred, markersize=7, colormap=(:viridis, 0.5))
    lines!(ax, r0:rs:r1, r0:rs:r1, color=:red)
    return fig
end


function plot_energy(e_pred::Vector{T}, e_true; labs=1:length(e_pred)) where T <: Vector{<:Real}
    r0 = minimum(e_true); r1 = maximum(e_true); rs = (r1-r0)/10

    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="E ref. [eV/atom]", ylabel="E pred. [eV/atom]")
   
    for k = 1:length(e_pred)
        scatter!(ax, e_true, e_pred[k], markersize=7, colormap=(:viridis, 0.5), label=labs[k])
    end
    lines!(ax, r0:rs:r1, r0:rs:r1, color=:red)
    axislegend(ax, position=:lt)
    return fig
end


# plot error #########################################################################
function plot_error(err_pred::Vector{Float64})
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1], xlabel="error [eV]", ylabel="count")
    hist!(ax, err_pred, bins=30)
    return fig
end

function plot_error(vref::Vector{T},
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

    fig = Figure(resolution=(1200,600))

    if logscl == false
        ax = Axis(fig[1, 1], xticks=(xoff, xticklabs), ylabel=ylab, title=ttl)
    else
        ax = Axis(fig[1, 1], xticks=(xoff, xticklabs), ylabel=ylab, title=ttl, yscale=log10)
        vecs = [abs.(v) .+ 1e-10 for v in vecs]
    end

    [hist!(ax, v, scale_to=xscl, color=(colors[i], 0.5), offset=xoff[i], bins=100, direction=:x) for (i,v) in enumerate(vecs)]

    return fig
end


# plot eigenspectrum ############################################################################
function plot_eigenspectrum(λ::Vector{Float64})
    fig = Figure(resolution=(500,500))
    ax = Axis(fig[1,1],
            title="Spectrum of gradient covariance matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λi)",
            xticks = 1:length(λ),
            yscale=log10,
            xgridvisible=false,
            ygridvisible=false)
    scatterlines!(ax, 1:length(λ), λ)
    return fig
end


function plot_eigenspectrum(λ::Vector{T}; labs=1:length(λ)) where T <: Vector{<:Real}
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Spectrum of matrix C",
            xlabel="index i",
            ylabel="eigenvalue (λi)",
            xticks = 1:length(λ),
            yscale=log10)

    for k = 1:length(λ)
        scatterlines!(ax, 1:length(λ[k]), λ[k], label=labs[k])
    end
    axislegend(ax)
    return fig
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



# plot wsd ######################################################################################
function plot_wsd(Cref::Matrix{<:Real}, C::Matrix{<:Real})
    d = size(Cref, 1)
    wsd = [WeightedSubspaceDistance(Cref, C, r) for r = 1:d]
    
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Weighted subspace distance",
            xlabel="dimension (r)",
            ylabel="WSD(r)",
            xticks = 1:d,
            # yscale=log10
            )

    scatterlines!(ax, 1:d, wsd)
    return fig, wsd
end


function plot_wsd(asr::T,
                  asis_arr::Vector{Vector{T}},
                  labs::Vector{String},
                  colors::Vector{Symbol};
                  markers::Vector{Symbol}=[:circle for i = 1:length(metvec)],
                  ls=[nothing for i = 1:length(asis_arr)]) where T <: Subspace
    
    d = size(asr.C, 1)

    with_theme(custom_theme) do
        fig = Figure(resolution=(600,600))
        ax = Axis(fig[1,1],
                title="Weighted subspace distance",
                xlabel="dimension (r)",
                ylabel="WSD(r)",
                xticks = 1:d,
                yscale=log10
                )

        for (i,asis) in enumerate(asis_arr)
            C_arr = [as.C for as in asis]
            wsd_arr = [[WeightedSubspaceDistance(asr.C, as.C, r) for r = 1:d] for as in asis]
            wmean = mean(wsd_arr)
            wmin = minimum(wsd_arr)
            wmax = maximum(wsd_arr)

            scatterlines!(ax, 1:d, wmean, color=colors[i], linestyle=ls[i], marker=markers[i], label=labs[i])
            band!(ax, 1:d, wmin, wmax, color=(colors[i], 0.3))

        end
        # axislegend(ax)
        return fig
    end
end


# plot cossim ######################################################################################
function plot_cossim(Cref::Matrix{<:Real}, C::Matrix{<:Real})
    d = size(Cref, 1)
    _, _, Wref = compute_eigenbasis(Cref)
    _, _, W = compute_eigenbasis(C)
    
    cossim = [Wref[:,i]'*W[:,i] for i = 1:d]
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Cosine similarity",
            xlabel="eigenvector (ϕi)",
            xticks = 1:d,
            ylabel="cos. sim.",
            # yscale=log10
            )

    scatterlines!(ax, 1:d, cossim)
    return fig, cossim
end

function plot_cossim(Cref::T, Cvec::Vector{T}, labs::Vector{Symbol}) where T <: Matrix{<:Real}
    d = size(Cref, 1)
    _, _, Wref = compute_eigenbasis(Cref)
    
    fig = Figure(resolution=(600,600))
    ax = Axis(fig[1,1],
            title="Cosine similarity",
            xlabel="eigenvector (ϕi)",
            xticks = 1:d,
            ylabel="cos. sim.",
            # yscale=log10
            )

    for (j,C) in enumerate(Cvec)
        _, _, W = compute_eigenbasis(C)
        cossim = [Wref[:,i]'*W[:,i] for i = 1:d]
        scatterlines!(ax, 1:d, cossim, label=labs[j])
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




# plot mcmc diagnostics ######################################################################################
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


function plot_mcmc_trace(postsamp::Vector{Vector{Float64}})
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



# plot IS diagnostics #######################################################################
function plot_IS_diag_2D_AS(mettype::String,
                            met::Vector,
                            C::Matrix, 
                            π::Distribution,
                            θsamp::Vector{Vector}
                            )

    # define 2D subspace
    as = compute_as(C, π, 2)

    # convert samples
    ysamp = (as.W1',) .* θsamp
    ymat = reduce(hcat, ysamp)

    fig = Figure(resolution=(550, 500))
    ax = Axis(fig[1,1], xlabel="y1", ylabel="y2", title=mettype)
    sc = scatter!(ax, ymat[1,:], ymat[2,:], markersize=7, color=met) # colormap=Reverse(:Blues)) 
    Colorbar(fig[1,1][1,2], sc)

    return fig
end


function plot_IS_diag_2D_AS(mettype::String,
                            metvec::Vector{<:Dict},
                            labs::Vector{String},
                            C::Matrix,
                            π::Distribution,
                            θsamp::Vector,
                            centers::Vector,
                            ttl::String;
                            logscl=false
                            )
    # define 2D subspace
    as = compute_as(C, π, 2)

    # convert samples
    ycent = (as.W1',) .* centers
    cmat = reduce(hcat, ycent)
    ysamp = (as.W1',) .* θsamp
    ymat = reduce(hcat, ysamp)

    # metric
    nmet = length(metvec)
    cmin = minimum([minimum(wi[mettype]) for wi in metvec])       # minimum value for colorbar
    cmax = maximum([maximum(wi[mettype]) for wi in metvec])       # maximum value for colorbar

    with_theme(custom_theme) do
        # plot diagnostic value vs. parameter value
        fig = Figure(resolution=(550*4, 500*1))
        ax = Vector{Axis}(undef, nmet)

        ax1 = [1, 1, 1, 1] # , 2, 2, 2, 2, 3, 3, 3, 3]
        ax2 = [1, 2, 3, 4] # , 1, 2, 3, 4, 1, 2, 3, 4]
        for i = 1:nmet
            ax[i] = Axis(fig[ax1[i],ax2[i]], xlabel="y1", ylabel="y2", title=labs[i])
            if logscl == false
                sc = scatter!(ax[i], ymat[1,:], ymat[2,:], markersize=7, color=metvec[i][mettype][1:length(θsamp)], colormap=Reverse(:Blues), colorrange=(cmin,cmax)) 
            else
                sc = scatter!(ax[i], ymat[1,:], ymat[2,:], markersize=7, color=log.(metvec[i][mettype][1:length(θsamp)]), colormap=Reverse(:Blues), colorrange=(log(cmin),log(cmax))) 
            end
            scatter!(ax[i], cmat[1,:], cmat[2,:], markersize=5, color=:red) # centers
            if i == length(metvec); Colorbar(fig[:,5], sc); end
        end
        Label(fig[0,:], text=ttl, fontsize=20)
        return fig
    end
end


function plot_IS_cdf(mettype::String,
                    metvec::Vector{<:Dict},
                    labs::Vector{String},
                    ylab::String,
                    ttl::String,
                    colors::Vector{Symbol};
                    limtype::Union{String, Vector}=nothing,
                    xticklab::Vector=nothing,
                    rev=false,
                    logscl=true,
                    markers::Vector{Symbol}=[:circle for i = 1:length(metvec)],
                    ls=[nothing for i = 1:length(asis_arr)]
                    )

    M = length(metvec)
    w_arr = [met_i[mettype] for met_i in metvec]

    # determine x limits
    if limtype == nothing
        wmin = 1e-8
        _, wmax = get_maxlimit(mettype, metvec)
    elseif limtype == "med"; wmin, wmax = get_medlimit(mettype, metvec);
    elseif limtype == "max"; wmin, wmax = get_maxlimit(mettype, metvec);
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
            lines!(ax, thresh, Pi, color=colors[k], linestyle=ls[k], linewidth=2)
            scatter!(ax, threshscat, Pis, color=colors[k], markersize=10, marker=markers[k], label=labs[k])
        end

        if mettype == "wdiag"
            vlines!(ax, 5.0, linestyle=:dash, color=:red, label="threshold")
        end
        # gridlines
        [vlines!(ax, t, color=(:black, 0.05)) for t in threshscat]

        # axislegend(ax) # position=:lb)
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


# plot eigenmodes #######################################################################
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


function plot_as_modes(W::Matrix,
                       ρθ::MvNormal,
                       ny::Int64,
                       r::Vector, # interatomic distances
                       B::Vector; # descriptors
                       scl=1) # scale (exaggeration factor)

    # adjust scale of covariance
    ρθ2 = MvNormal(ρθ.μ, scl * ρθ.Σ)

    # adjust indexing
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

        # compute sampling densities
        π_y = compute_marginal(W1, ρθ2)
        π_z = compute_marginal(W2, ρθ2)

        # draw samples
        ysamp = [rand(π_y) for i = 1:ny]
        θy = [W1*y + W2*π_z.μ for y in ysamp]

        # compute energies
        energies = [Bi' * βi for Bi in B, βi in θy]

        # define axis
        if i < d/2 + 1
            ax[i] = Axis(fig[i,1], xlabel="r (Å)", ylabel="E (eV)", title="mode $i") # yticks=-0.006:0.002:0.00, limits=(1, 5, -0.008, 0.002)
        elseif i == Int(d/2)
            ax[i] = Axis(fig[i,1], xlabel="r (Å)", ylabel="E (eV)", title="mode $i")
        elseif i == d 
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="r (Å)", ylabel="E (eV)", title="mode $i")
        else
            ax[i] = Axis(fig[i-Int(d/2),2], xlabel="r (Å)", ylabel="E (eV)", title="mode $i")
        end
            
        # plot
        [lines!(ax[i], r, energies[:,j]) for j in 1:ny] # color=(:blue, 0.2)
    end
    fig
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


# plot pairwise (interatomic) potential ###################################################
function plot_pairwise_potential(θsamp::Vector{T},
                                 r::Vector{<:Real},
                                 B::Vector{T}) where T <: Vector{<:Real}


    with_theme(custom_theme) do
        f = Figure(resolution=(650, 650))
        ax = Axis(f[1, 1], xlabel="interatomic distance r (Å)",
        ylabel="energies E (eV)",
        title="Pairwise energy curve for samples from ρ(θ)")

        _plot_pairwise_potential(θsamp, r, B, ax, lab="samples")
        axislegend(ax, position=:rb)
        return f
    end
end

function plot_pairwise_potential(θvec::Vector{Vector{T}},
                                r::Vector{<:Real},
                                B::Vector{T},
                                colors::Vector{Symbol},
                                labs::Vector{String},
                                ) where T <: Vector{<:Real}

    nplot = length(θvec)

    with_theme(custom_theme) do
        f = Figure(resolution=(650, 650))
        # ax = Vector{Axis}(undef, nplot)
        ax = Axis(f[1, 1], xlabel="interatomic distance r (Å)",
                    ylabel="energies E (eV)")

        for i = 1:nplot
             _plot_pairwise_potential(θvec[i], r, B, ax, lab=labs[i], col=colors[i])
        end
        axislegend(ax, position=:rb)
        return f
    end
end


function _plot_pairwise_potential(θsamp::Vector{T},
                                 r::Vector{<:Real},
                                 B::Vector{T},
                                 ax::Axis;
                                 lab::String="",
                                 col=:grey) where T <: Vector{<:Real}

    energies = [Bi' * θi for Bi in B, θi in θsamp] 

    [lines!(ax, r, energies[:, i], color=(col, 0.3), linewidth=1) for i = 1:(length(θsamp)-1)]
    lines!(ax, r, energies[:, end], color=(col, 0.3), linewidth=1, label=lab)
end