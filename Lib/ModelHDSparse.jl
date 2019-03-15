#=====================
    SDMs model 0.1
======================#

using SparseArrays

module HDSparse
    import SparseArrays
    import SparseArrays.sparsevec

    BITFORMAT = Int64

    function RandomGenerator(n, w)
        # generates random vector of form [0 0 0 ... 0 1 0 ... 0 1 0 0 0 ...]
        rand(1:n, w) |> sort |>
            (ri -> sparsevec(ri, [1 for i=1:length(ri)]))
    end

    function RandomGenerator(dict)
        RandomGenerator(dict[:n], dict[:w])
    end

    function composition(vectorHD::Vector{SparseArrays.SparseVector{BITFORMAT,
                                                                    BITFORMAT}})
        idces = unique(vcat(map(v -> v.nzind, vectorHD)...))
        sparsevec(idces, map(i -> BITFORMAT(1), idces))
    end

    function cosineSimilarity(ri1::SparseArrays.SparseVector{BITFORMAT,
                                                             BITFORMAT},
                              ri2::SparseArrays.SparseVector{BITFORMAT,
                                                             BITFORMAT})
        sparseVectorMultiplication(ri1, ri2) =
            filter(x -> x in ri1.nzind, ri2.nzind) |>
                    length

        sparseVectorMultiplication(ri1, ri2) /
            (length(ri2.nzval) * length(ri2.nzval))^.5
    end

    function encodeOnTheFly(dic1DSparseRepres, modelEncoding, wrdOrTag)
        if haskey(dic1DSparseRepres, wrdOrTag)
            dic1DSparseRepres[wrdOrTag].nzind
        else
            randSDM = RandomGenerator(modelEncoding[:n], modelEncoding[:w])
            dic1DSparseRepres[wrdOrTag] = randSDM
            randSDM.nzind
        end
    end

end

export HDSparse
