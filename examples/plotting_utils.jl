function plot_val_metric(val_type::String, nsamp_arr::Vector{Int64}, val_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String; logscl=true)
    fig = Figure(resolution=(800, 500))
    if logscl == true
        ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel=ylab,
        title=ttl, xscale=log10, yscale=log10)   
    else
        ax = Axis(fig[1,1], xlabel="number of samples (n)", ylabel=ylab,
        title=ttl, xscale=log10) 
    end
    for (val_i, lab_i) in zip(val_tup, lab_tup)
        scatterlines!(ax, nsamp_arr, abs.(val_i[val_type]), label=lab_i)
    end
    axislegend(ax, position=:ct)
    fig
end

function plot_IS_diagnostic(met_type::String, met_tup::Tuple, lab_tup::Tuple, ylab::String, ttl::String; logscl=false)
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    nIS = length(w_arr)

    cmin = minimum([minimum(wi) for wi in w_arr])       # minimum value for colorbar
    cmax = maximum([maximum(wi) for wi in w_arr])       # maximum value for colorbar

    θsamp = met_tup[1]["θsamp"][1]
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
        
        ax[i] = Axis(fig[1,i][1,1], xlabel="θ1", ylabel="θ2",
        title=lab_tup[i]) 
        # plot 2D scatterplot and colorbar
        sc = scatter!(ax[i], θmat[1,:], θmat[2,:], markersize=7, color=wi, colormap=:RdBu, colorrange=(cmin,cmax)) 
        if i == nIS
            Colorbar(fig[1,i][1,2], sc, label=ylab)
        end  
    end
    Label(fig[0,:], text=ttl, fontsize=20)
    return fig
end


function plot_IS_samples(πg0::Gibbs, met_type::String, met_tup::Tuple, lab_tup::Tuple, dist::Tuple, ttl::String, xlim::Vector, quadpts::Tuple; rev=false, nsamp=100, thresh=:none)
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    θsamp = met_tup[1]["θsamp"][1]
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

        ax[j] = Axis(fig[1,j],  xlabel="x", ylabel="pdf(x)", limits=(xlim[1], xlim[2], -0.02, 0.65))
        xplot = LinRange(xlim[1], xlim[2], 1000)

        # plot first nsamp samples 
        if m != 0
            for (i, θi) in zip(1:m, θsamp[ids])
                πg = Gibbs(πg0, β=1.0, θ=θi)
                if i == 1
                    lines!(ax[j], xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:steelblue, 0.1), label="prop: $(propsamp)%") 
                else
                    lines!(ax[j], xplot, updf.(πg, xplot) ./ normconst(πg, quadpts[1], quadpts[2]), color=(:steelblue, 0.1)) 
                end
            end
        else
            lines!(ax[j], xplot, zeros(length(xplot)), color=(:steelblue, 0.001), label="prop: 0%") 
        end
        # plot biasing distribution
        try
        lines!(ax[j], xplot, pdf.(dist[j], xplot), color=:red, label=lab_tup[j])
        catch
            lines!(ax[j], xplot, updf.(dist[j], xplot) ./ normconst(dist[j], quadpts[1], quadpts[2]), color=:red, label=lab_tup[j])
        end
        axislegend(ax[j])
    end

    if thresh == :none 
        ttl *= " (top $nsamp samples)"
    else
        th = round(thresh, digits=1)
        ttl *= " (threshold = $th)"
    end
    Label(fig[0,:], text=ttl, fontsize=20) # title
    return fig
end


function compute_invcdf(w::Vector{Float64}; thresh::Union{Nothing,Vector{Float64}}=nothing)
    
    if thresh == nothing
        thresh = exp.(LinRange(log.(minimum(w)), log.(maximum(w)), 10))
    end

    P = Vector{Float64}(undef, length(thresh))
    for (i,t) in zip(1:length(thresh), thresh)
         P[i] = length(findall(x -> x > t, w)) / length(w)
    end

    return P
end


function get_medlimit(met_type::String, BD::String, nsamp_arr::Vector{Int64}, nrepl_arr::Vector{Int64}; β=nothing)
    M = length(nsamp_arr)
    wmax = Vector{Float64}(undef, M)
    wmin = Vector{Float64}(undef, M)

    for (j, nsamp, nrepl) in zip(1:M, nsamp_arr, nrepl_arr)
        if β == nothing
            metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
        else
            metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"][β]
        end
        wmax[j] = maximum(metrics[met_type][1])
        wmin[j] = minimum(metrics[met_type][1])
    end
    return median(wmin), median(wmax)
end


function get_medlimit(met_type::String, met_tup::Tuple)
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    
    wmax = [maximum(wi) for wi in w_arr]
    wmin = [minimum(wi) for wi in w_arr]
    
    return median(wmin), median(wmax)
end


function get_maxlimit(met_type::String, BD::String, nsamp_arr::Vector{Int64}, nrepl_arr::Vector{Int64})
    M = length(nsamp_arr)
    wmax = Vector{Float64}(undef, M)
    wmin = Vector{Float64}(undef, M)

    for (j, nsamp, nrepl) in zip(1:M, nsamp_arr, nrepl_arr)
        if β == nothing
            metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
        else
            metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"][β]
        end
        wmax[j] = maximum(metrics[met_type][1])
        wmin[j] = minimum(metrics[met_type][1])
    end
    return minimum(wmin), maximum(wmax)
end


function get_maxlimit(met_type::String, met_tup::Tuple)
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    
    wmax = [maximum(wi) for wi in w_arr]
    wmin = [minimum(wi) for wi in w_arr]
    
    return minimum(wmin), maximum(wmax)
end


function plot_IS_cdf(met_type::String, BD::String, nsamp_arr::Vector{Int64}, nrepl_arr::Vector{Int64}, ttl::String; β=nothing, limittype="med")
    M = length(nsamp_arr)
    if limittype == "med"; wmin, wmax = get_medlimit(met_type, BD, nsamp_arr, nrepl_arr; β=β);
    elseif limittype == "max"; wmin, wmax = get_maxlimit(met_type, BD, nsamp_arr, nrepl_arr; β=β); end
    thresh = exp.(LinRange(log.(wmin), log.(wmax), 50))

    fig = Figure(resolution = (450, 500))
    ax = Axis(fig[1,1], xlabel="threshold (t)", ylabel="P[w > t]", xscale=log10, title=ttl)

    
    for (k, nsamp, nrepl) in zip(1:M, nsamp_arr, nrepl_arr)
        if β == nothing
            metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"]
        else
            metrics = JLD.load("data1/DW1D_$(BD)_nsamp={$nsamp}_nrepl={$nrepl}.jld")["metrics"][β]
        end
        w = metrics[met_type][1]
        Pi = compute_invcdf(w; thresh=thresh)
        # scatter!(ax, thresh, Pi, color=k, colormap=:Blues, colorrange=(0,M), label="n = $nsamp")
        lines!(ax, thresh, Pi, color=k, colormap=:Blues, colorrange=(0,M), label="n = $nsamp")
    end

    axislegend(ax)
    return fig
end


function plot_IS_cdf(met_type::String, met_tup::Tuple, lab_tup::Tuple, ttl::String; limittype="med")
    M = length(met_tup)
    w_arr = [met_i[met_type][1] for met_i in met_tup]
    if limittype == "med"; wmin, wmax = get_medlimit(met_type, met_tup);
    elseif limittype == "max"; wmin, wmax = get_maxlimit(met_type, met_tup); end
    thresh = exp.(LinRange(log.(wmin), log.(wmax), 50))

    fig = Figure(resolution = (450, 500))
    ax = Axis(fig[1,1], xlabel="threshold (t)", ylabel="P[w > t]", xscale=log10, title=ttl)

    
    for (k, wi) in zip(1:M, w_arr)
        Pi = compute_invcdf(wi; thresh=thresh)
        lines!(ax, thresh, Pi, label=lab_tup[k])
    end

    if met_type == "wdiag"
        vlines!(ax, 5.0, linestyle=:dash, color=:red, label="threshold")
    end

    axislegend(ax) # position=:lb)
    return fig
end
