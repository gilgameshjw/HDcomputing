using SparseArrays
using StatsBase
using ProgressMeter
using Compat, Random, Distributions
using LinearAlgebra
using PlotlyJS
using Blink
using WebIO
using ORCA

include("Lib/ModelDense.jl")
include("Lib/ModelHDSparse.jl")
include("Lib/ModelHDBinary.jl")


function generateRun(nbVectors,
                     nbNeighbours,
                     vRndGenFn,
                     compositionFn,
                     similFn)

    compositionScores, randomScores, singleRetrievalScores =
                            Dict(), Dict(), Dict()
    randVectors = [map(x -> vRndGenFn(1), 1:nbVectors)]
    for i=1:nbNeighbours-1
        println(i)
        compositionScores[i] = (length(randVectors) == 1) ?
            map((v, w) -> similFn(v, w), randVectors[1], randVectors[end]) :
            map((v, w) -> similFn(v, w), randVectors[end-1], randVectors[end])
        singleRetrievalScores[i] =
            map((v, w) -> similFn(v, w), randVectors[1], randVectors[end])
        rdnscores = Float64[]
        for i=1:length(randVectors[end])
            for j=1:2 # i we don't span all combinations
                idx = max((i + j) % nbVectors, 1)
                push!(rdnscores,
                      similFn(randVectors[end][i], randVectors[end][idx]))
            end
        end
        randomScores[i] = rdnscores
        push!(randVectors,
              map(v -> compositionFn([v, vRndGenFn(1)]), randVectors[end]))
    end
    map(i -> mean_and_std(compositionScores[i]), 1:nbNeighbours-1),
        map(i -> mean_and_std(singleRetrievalScores[i]), 1:nbNeighbours-1),
            map(i -> mean_and_std(randomScores[i]), 1:nbNeighbours-1)
end

function makeTrace(x, y, error, dict)
    if error != Nothing
        error_y=attr(type="data",color=dict[:color],symmetric=true,array=error, opacity=0.75)
    else
        error_y=""
    end
    scatter(;x=x, y=y, error_y=error_y,
            name=dict[:label],
            mode=dict[:markers],
            marker=attr(color=dict[:color], opacity=0.25,
                        line_color=dict[:color],
                        line_width=1, size=10, symbol=dict[:symbol]),
            line=attr(width=1))
end

function doAnalysis(dicParameters)
    name = dicParameters[:title]
    plotsOverview, plotsDeltas = [], []
    for (i, dicP) in enumerate(dicParameters[:params])
        color = dicParameters[:colors][i]
        label = mapreduce(x->string(x, " "), string,
                          [string(kv[1], ": ", kv[2]) for kv in collect(dicP)])
        fctGenerator(x) = dicParameters[:fct](dicP)
        fctComposition = dicParameters[:composition]
        fctSimilarity = dicParameters[:similarity]
        nbVectors, nbNeighbours =
                dicParameters[:nbVectors], dicParameters[:nbNeighbours]
        compositionScores,  singleRetrievalScores, randomScores =
                                        generateRun(nbVectors,
                                                    nbNeighbours,
                                                    fctGenerator,
                                                    fctComposition,
                                                    fctSimilarity)

        dict = Dict(:label => string("Set_1 VS Set_N: ", "<b>", label,"</b>"),
                    :color => color, :symbol => "diamond", :markers => "markers")
        tr1 = makeTrace(1:nbNeighbours-1,
                        [s[1] for s in singleRetrievalScores],
                        [s[2] for s in singleRetrievalScores], dict)
        dict = Dict(:label => string("Set_1 VS Set_N: ","<b>", label,"</b>"),
                    :color => color, :symbol => "diamond", :markers => "lines+markers")
        ttr1 = makeTrace(2:nbNeighbours-1,
                         map((x,y) -> (x[1]-y[1]-(x[2]+y[2])) / (x[2]+y[2]),
                             singleRetrievalScores[2:end-1],
                             singleRetrievalScores[3:end]),
                         Nothing, dict)
        dict = Dict(:label => string("Set_N+1 VS Set_N"), #, label,"</b>"),
                    :color => color, :symbol => "square", :markers => "markers")
        tr2 = makeTrace(1:nbNeighbours-1,
                        [s[1] for s in compositionScores],
                        [s[2] for s in compositionScores], dict)

        dict = Dict(:label => string("Set_N+1 VS Set_N", "<b>", label,"</b>"),
                    :color => color, :symbol => "square", :markers => "lines+markers")
        ttr2 = makeTrace(2:nbNeighbours-1,
                         map((x,y) -> y[1]-x[1]-(x[2]+y[2]),
                             compositionScores[2:end-1],
                             compositionScores[3:end]),
                         Nothing, dict)

        dict = Dict(:label => string("Random VS Random"), :color => color,
                    :symbol => "circle", :markers => "markers")
        tr3 = makeTrace(1:nbNeighbours-1,
                        [s[1] for s in randomScores],
                        [s[2] for s in randomScores], dict)
        push!(plotsOverview, tr1, tr2, tr3)
        push!(plotsDeltas, ttr1, ttr2)
    end

    layout1 = Layout(;title = name,
                     xaxis=attr(title="<b> Nb of Elements in Set </b>", showgrid=false, zeroline=false),
                     yaxis=attr(title="<b> Averaged Similarity </b>", zeroline=false),
                     shapes=[hline(0)],
                     font=attr(size=16))
    layout2 = Layout(;title = name,
                      xaxis=attr(title="<b> Nb of Elements in Set </b>", showgrid=false, zeroline=false),
                      yaxis=attr(title="<b> (Sim_N+1-Sim_N + std_N+1+std_N) / (std_N+1+std_N) </b>" ,
                                 zeroline=false),
                      shapes=[hline(0)],
                      font=attr(size=16))
    Dict(:plotOverview => plot(vcat(plotsOverview...), layout1),
         :plotDeltas => plot(vcat(plotsDeltas...), layout2))
end


#############
# HD Sparse #
#############

params = map((n, w) -> Dict(:n => n, :w => w),
            [1000, 10000, 100000, 1000000], [20, 20, 20, 20])
dicParameters = Dict(:params => params,
                     :fct => HDSparse.RandomGenerator,
                     :composition => HDSparse.composition,
                     :similarity => HDSparse.cosineSimilarity,
                     :nbVectors => 1000,
                     :nbNeighbours => 40,
                     :title => "<b> HD Sparse Vectors </b>",
                     :colors => ["blue", "green", "brown", "magenta", "red"])
p = dicParameters |> doAnalysis

figFileName = string("Results/HDSparseFig1Overview.pdf")
savefig(p[:plotOverview], figFileName)
figFileName = string("Results/HDSparseFig1Deltas.pdf")
savefig(p[:plotDeltas], figFileName)
figFileName = string("Results/HDSparseFig1Overview.jpeg")
savefig(p[:plotOverview], figFileName)
figFileName = string("Results/HDSparseFig1Deltas.jpeg")
savefig(p[:plotDeltas], figFileName)

#############
# HD Binary #
#############

params = map(n -> Dict(:n => n), [500, 5000, 10000, 15000])
dicParameters = Dict(:params => params,
                     :fct => HDBinary.RandomGenerator,
                     :composition => HDBinary.composition,
                     :similarity => HDBinary.cosineSimilarity,
                     :nbVectors => 1000,
                     :nbNeighbours => 40,
                     :title => "HD Dense Binary Vectors",
                     :colors => ["blue", "green", "brown", "magenta", "red"])
p = dicParameters |> doAnalysis

figFileName = string("Results/HDBinaryFig1Overview.pdf")
savefig(p[:plotOverview], figFileName)
figFileName = string("Results/HDBinaryFig1Deltas.pdf")
savefig(p[:plotDeltas], figFileName)
figFileName = string("Results/HDBinaryFig1Overview.jpeg")
savefig(p[:plotOverview], figFileName)
figFileName = string("Results/HDBinaryFig1Deltas.jpeg")
savefig(p[:plotDeltas], figFileName)

#########
# Dense #
#########

params = map(n -> Dict(:n => n), [50, 500,  1000, 5000])
dicParameters = Dict(:params => params,
                     :fct => Dense.RandomGenerator,
                     :composition => Dense.composition,
                     :similarity => Dense.cosineSimilarity,
                     :nbVectors => 1000,
                     :nbNeighbours => 40,
                     :title => "Dense Vectors",
                     :colors => ["blue", "green", "brown", "magenta", "red"])
p = dicParameters |> doAnalysis

figFileName = string("Results/HDDenseFig1Overview.pdf")
savefig(p[:plotOverview], figFileName)
figFileName = string("Results/HDDenseFig1Deltas.pdf")
savefig(p[:plotDeltas], figFileName)
figFileName = string("Results/HDDenseFig1Overview.jpeg")
savefig(p[:plotOverview], figFileName)
figFileName = string("Results/HDDenseFig1Deltas.jpeg")
savefig(p[:plotDeltas], figFileName)
