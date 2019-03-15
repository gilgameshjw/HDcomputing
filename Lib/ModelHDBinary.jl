#=====================
    HD binary model 0.1
======================#

using SparseArrays
using StatsBase

module HDBinary
    import SparseArrays
    import SparseArrays.sparsevec
    import StatsBase

    BITFORMAT = Int32

    function RandomGenerator(n::Int64)
        # generates binary vector of form [0 1 0 1 1 0 0 0 ... 1 0 1 0 0 1 1]
        v = zeros(BITFORMAT, n)
        for i in StatsBase.sample(1:n, Int64(n/2), replace = false)
            v[i] = BITFORMAT(1)
        end
        v
    end

    function RandomGenerator(dict::Dict{Symbol, Int64})
        RandomGenerator(dict[:n])
    end

    function composition(vectorHD::Vector{Vector{BITFORMAT}})
        vv = hcat(vectorHD...)
        N, L = size(vv)
        toBinary(x) =   if x == .5
                            rand([0, 1])
                        elseif x > .5
                            1
                        elseif x < .5
                            0
                        end
        map(i -> toBinary((vv[i, :] |> sum) / L) |> BITFORMAT, 1:N)
    end

    # std cosine sim
    function cosineSimilarity(hdv1::Vector{BITFORMAT}, hdv2::Vector{BITFORMAT})
        1. - 2*(sum(map((x,y) -> xor(x, y), hdv1, hdv2)) / length(hdv1))
        #sum(map((x,y) -> xor(x, y), hdv1, hdv2)) / length(hdv1)
    end

end

export HDBinary
