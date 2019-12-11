module MyMPS
include("tsops.jl")
using .MyTensorOps
export initmps, 𝑿, 𝒀, 𝒁, 𝑯

#=

Convention for MPS

     i₂
     |
     |
i₁---⊕---i₃

=#
mutable struct MPS
    data::Vector{Array{Number, 3}}
end

#=

Convention for OnebodyGate -- Gate₁

     i₁
     |
     ⊕
     |
     i₂

=#
struct Gate₁
    op::Array{𝕋, 2} where {𝕋<:Complex{Float64}}
end

#=

Convention for TwobodyGate -- Gate₂

     i₁ i₂
     |  |
    ------
    ------
     |  |
     i₃ i₄

=#
struct Gate₂
    i::Array{Int64, 1}
    op::Array{𝕋, 4} where {𝕋<:Complex{Float64}}
end
function initmps(basis::AbstractArray)
    d = [x==0 ? reshape([1,0], 1, 2, 1) : reshape([0,1], 1, 2, 1) for x in basis]
    return MPS(d)
end

# 1:2:1 contract! with 2:2 -> 1:1:2, so we need to reshape to 1:2:1 and reset the tensor
function 𝑿(sys::MPS, i::Int64)
    sys.data[i] = permutedims(contract!(sys.data[i], [0 1;1 0], (2,), (2,)), [1,3,2])
end

function 𝒀(sys::MPS, i::Int64)
    sys.data[i] = permutedims(contract!(sys.data[i], [0 -im;im 0], (2,), (2,)), [1,3,2])
end

function 𝒁(sys::MPS, i::Int64)
    sys.data[i] = permutedims(contract!(sys.data[i], [1 0;0 -1], (2,), (2,)), [1,3,2])
end

function 𝑯(sys::MPS, i::Int64)
    sys.data[i] = permutedims(contract!(sys.data[i], 1/sqrt(2)*[1 1;1 -1], (2,), (2,)), [1,3,2])
end

import LinearAlgebra:svd, Diagonal

function tsvd(a::AbstractArray{𝕋, ℕ}, axes::Tuple) where {𝕋, ℕ}
    a = Number.(a)
    sizeA = size(a)
    laxes = filter(x -> !(x in axes), collect(1:ℕ)) # not in the axes, permute to left side
    raxes = [axes...] # in the axes, permute to right side

    lsize = [sizeA[x] for x in laxes] # store original size for reshaping after the svd is done
    rsize = [sizeA[x] for x in raxes]

    # permute A, reshape to matrix then do svd
    u, s, vt = svd(reshape(permutedims(a, [laxes;raxes]), prod(lsize), prod(rsize)))
    u = u * sqrt(Diagonal(s)) # s will be absorbed into u and vt
    vt = vt * sqrt(Diagonal(s))

    # reshape to original order
    dim = length(s)
    return (reshape(u, lsize..., dim), reshape(vt, dim, rsize...))
end

# only can swap closed tensor
function swap_closed(sys::MPS, i₁::Int64, i₂::Int64)
    i₁, i₂ = min(i₁, i₂), max(i₁, i₂)
    if i₂ - i₁ != 1 error("parameter wrong") end
    contract!(sys.data[i₁], sys.data[i₂], (3,), (1,))
    res2g = contract!(sys.data[i₁], sys.data[i₂], (3,), (1,))
    swapg = reshape([1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1], 2, 2, 2, 2)
    tmp = contract!(res2g, swapg, (2, 3), (3, 4))
    u, vt = tsvd(reshape(tmp, 1, 2, 2, 1), (2, 4)) # ERROR!!!!
    # u is okay with convention, vt need to permute
    vt = permutedims(vt, (1,3,2))
    sys.data[i₁] = u
    sys.data[i₂] = vt
    return (u, vt)
end

# get one controlled single-qubit gate
function controlled(gate::AbstractArray)
    if size(gate) !== (2,2) error("parameter wrong") end
    # generate |0><0|⨂Identity
    a = contract!([1 0;0 0], [1 0;0 1], (), ())
    # generate |1><1|⨂OnebodyGate
    b = contract!([0 0;0 1], gate, (), ())
    return permutedims(a+b, [1,3,2,4])
end

function CNOT(sys::MPS, con::Int64, op::Int64)
    g₂ = controlled([0 1;1 0])
    sites = contract!(sys.data[con], sys.data[op], (3,), (1,))
    sites = contract!(sites, g₂, (2,3), (3,4))
    u, vt = tsvd(sites, (2,4))
    println("--", size(u))
    println("||", size(vt))
    vt = permutedims(vt, [1,3,2])
    sys.data[con] = u
    sys.data[op] = vt
    return (u, vt)
end

# not finished
function projection(sys::MPS, proj::MPS)
    length(sys.data) != length(proj.data) && error("system mps and projection mps's dims must be equal")

end

# not finished
function measure(sys::MPS, i::Int64)
    permutedims(contract!(sys.data[i], [1 0;0 0], (2,), (2,)), [1,3,2])
end

end
