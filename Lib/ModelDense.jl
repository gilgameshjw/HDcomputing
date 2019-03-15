#=====================
    Dense model 0.1
======================#

using SparseArrays

module Dense

    BITFORMAT = Float64

    function RandomGenerator(n::Int64, denseType="uniform")
        if denseType == "uniform"
            return rand(BITFORMAT, n) - map(x -> Float16(.5), 1:n)
        elseif denseType == "gaussian"
            d = Normal()
            return rand(d, n)
        end
    end

    function RandomGenerator(dict)
        RandomGenerator(dict[:n], getkey(dict, :denseType, "uniform"))
    end

    normalise(v::Vector{BITFORMAT}) = v / (v'*v)^.5

    function composition(vectorHD::Vector{Vector{BITFORMAT}})
    vv = hcat(vectorHD...)
    N, L = size(vv)
    map(i -> vv[i, :] |> sum, 1:N) |> normalise
    end

    # std cosine sim
    function cosineSimilarity(v1::Vector{BITFORMAT}, v2::Vector{BITFORMAT})
        v1' * v2 / ((v1' * v1) * (v2' * v2))^.5 |> BITFORMAT
    end

end

export Dense
