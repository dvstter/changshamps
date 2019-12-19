module MyVQC

export contract
export MPS, QuantumCircuit, OnebodyGate, TwobodyGate, initmps, initcircuit, add_gate
export pauli_x, pauli_y, pauli_z, hadamard, cnot
export apply!, 𝑿!, 𝒀!, 𝒁!, 𝑯!, projection, projection_all, measure

#=
    Basic functions to support tensor network computation.
=#
function contract(a::AbstractArray{Ta, Na}, b::AbstractArray{Tb, Nb}, dima::Tuple, dimb::Tuple) where {Ta, Tb, Na, Nb}
    if [size(a)[x] for x in dima] != [size(b)[x] for x in dimb] error("size is wrong") end
    if length(dima) == 0
        a_size = [size(a)...]
        b_size = [size(b)...]
        ra = reshape(a, prod(a_size), 1)
        rb = reshape(b, 1, prod(b_size))
        return reshape(ra*rb, Tuple([a_size; b_size]))
    end

    a_left = filter(x -> !(x in dima), [1:Na...])
    a_right = [dima...]
    a_perm = [a_left; a_right]
    a_reshape_size = [size(a)[x] for x in a_left]
    a_left_len = prod(a_reshape_size)
    a_right_len = prod([size(a)[x] for x in a_right])

    b_left = [dimb...]
    b_right = filter(x -> !(x in dimb), [1:Nb...])
    b_perm = [b_left; b_right]
    b_left_len = prod([size(b)[x] for x in b_left])
    b_reshape_size = [size(b)[x] for x in b_right]
    b_right_len = prod(b_reshape_size)

    am = permutedims(a, a_perm)
    bm = permutedims(b, b_perm)

    # NOTES: I have spent alot of time on these codes
    ra = reshape(am, (a_left_len, a_right_len))
    rb = reshape(bm, (b_left_len, b_right_len))

    return Number.(reshape(ra*rb, Tuple([a_reshape_size; b_reshape_size])))
end

import LinearAlgebra:svd, Diagonal
using LinearAlgebra.LAPACK: gesvd!, gesdd!

function stable_svd!(a::AbstractArray{𝕋, 2}) where 𝕋
    try
        return gesdd!('S', copy(a))
    catch
        return gesvd!('S', 'S', a)
    end
end

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
    # NOTES: I have spend alot of time on this line of code
    v = transpose(vt)

    # reshape to original order
    dim = length(s)
    return (reshape(u, lsize..., dim), reshape(v, dim, rsize...))
end

#=
    Quantum Structures.
    For conventions, please refer to the original implementation
=#
struct MPS
    data::Vector{Array{Number, 3}}
end

struct OnebodyGate
    index::Int64
    operation::Array{Number, 2}
end

struct TwobodyGate
    indexes::Array{Int64, 1}
    operation::Array{Number, 4}
end

struct QuantumCircuit
    gates::Array{Union{OnebodyGate, TwobodyGate}, 1}
end

#=
    Quantum circuit functions defined here.
=#

function initmps(basis::AbstractArray)
    res = []
    for x in basis
        t = []
        if x == 0
            t = reshape([1,0], 1, 2, 1)
        elseif x == 1
            t = reshape([0,1], 1, 2, 1)
        else
            x = rand()
            y = sqrt(1-x)
            t = reshape([sqrt(x), y], 1, 2, 1)
        end
        push!(res, Number.(t))
    end
    return MPS(res)
end

function initcircuit()
    return QuantumCircuit([])
end

function add_gate(circuit::QuantumCircuit, gate::Union{OnebodyGate, TwobodyGate})
    push!(circuit.gates, gate)
end

# get one controlled single-qubit gate
# note that this gate can only be applied on two adjacent site
function __reverse_gate(gate::AbstractArray)
    size(gate) != (2,2,2,2) && error("parameter wrong")
    return permutedims(gate, [2,1,4,3])
end

# get one swap gate
# note that this gate can only be applied on two adjacent site
function __swap_gate()
    cnot₁ = __controlled_gate([0 1;1 0])
    cnot₂ = deepcopy(cnot₁)
    cnot₃ = deepcopy(cnot₁)
    g₂ = contract(cnot₁, cnot₂, (3,4), (2,1))
    return contract(g₂, cnot₃, (4,3), (1,2))
end

# get one controlled single-qubit gate
# note that this gate can only be applied on two adjacent site
function __controlled_gate(gate::AbstractArray)
    size(gate) !== (2,2) && error("parameter wrong")
    # generate |0><0|⨂Identity
    a = contract([1 0;0 0], [1 0;0 1], (), ())
    # generate |1><1|⨂OnebodyGate
    b = contract([0 0;0 1], gate, (), ())
    return permutedims(a+b, [1,3,2,4])
end

function __adjacent_SWAP(sys::MPS, i₁::Int64, i₂::Int64)
    abs(i₂ - i₁) != 1 && error("parameter wrong")
    i₁ > i₂ && begin i₁, i₂ = i₂, i₁ end
    g₂ = __swap_gate()
    sites = contract(sys.data[i₁], sys.data[i₂], (3,), (1,))
    sites = contract(sites, g₂, (2,3), (3,4))
    u, vt = tsvd(sites, (4,2))
    sys.data[i₁] = u
    sys.data[i₂] = vt
    return (u, vt)
end

# frequently-used quantum gates
pauli_x(i::Int64) = OnebodyGate(i, [0 1;1 0])
pauli_y(i::Int64) = OnebodyGate(i, [0 -im;im 0])
pauli_z(i::Int64) = OnebodyGate(i, [1 0;0 -1])
hadamard(i::Int64) = OnebodyGate(i, 1/sqrt(2)*[1 1;1 -1])
cnot(con::Int64, op::Int64) = TwobodyGate([con, op], __controlled_gate([0 1;1 0]))

function __apply_onebody_gate(sys::MPS, gate::OnebodyGate)
    sys.data[gate.index] = permutedims(contract(sys.data[gate.index], gate.operation, (2,), (2,)), [1,3,2])
end

function __apply_twobody_gate(sys::MPS, gate::TwobodyGate)
    idx₁ = gate.indexes[1]
    idx₂ = gate.indexes[2]
    g₂ = gate.operation
    idx₁ > idx₂ && begin idx₁, idx₂ = idx₂, idx₁; g₂ = __reverse_gate(g₂) end
    sorder = [(x, x+1) for x in idx₁:idx₂-2]
    revorder = reverse(sorder)
    for (x, y) in sorder
        __adjacent_SWAP(sys, x, y)
    end
    sites = contract(sys.data[idx₂-1], sys.data[idx₂], (3,), (1,))
    sites = contract(sites, g₂, (2,3), (3,4))
    u, vt = tsvd(sites, (4,2))
    sys.data[idx₂-1] = u
    sys.data[idx₂] = vt
    for (x, y) in revorder
        __adjacent_SWAP(sys, x, y)
    end
end

function apply!(sys::MPS, circuit::QuantumCircuit)
    for x in circuit.gates
        isa(x, OnebodyGate) && __apply_onebody_gate(sys, x)
        isa(x, TwobodyGate) && __apply_twobody_gate(sys, x)
    end
    return sys
end

# directly apply gate on the mps
𝑿!(sys::MPS, index::Int64) = __apply_onebody_gate(sys, pauli_x(index))
𝒀!(sys::MPS, index::Int64) = __apply_onebody_gate(sys, pauli_y(index))
𝒁!(sys::MPS, index::Int64) = __apply_onebody_gate(sys, pauli_z(index))
𝑯!(sys::MPS, index::Int64) = __apply_onebody_gate(sys, hadamard(index))

function projection(sys::MPS, basis::AbstractArray, output::Bool=false)
    length(basis) != length(sys.data) && error("parameter wrong, projection and sys's dims not equal")
    proj = initmps(basis)
    nqubits = length(sys.data)
    t = contract(sys.data[nqubits], proj.data[nqubits], (2,), (2,))
    for x in (nqubits-1):-1:1
        t₀ = contract(sys.data[x], proj.data[x], (2,), (2,))
        t = contract(t, t₀, (1,3), (2,4))
        # these three methods should be contract to one method, but i don't know how to yet
        t = permutedims(t, (2,1,3,4))
        t = permutedims(t, (1,2,4,3))
        t = permutedims(t, (4,2,3,1))
    end
    res = reshape(t, 1)[1]
    output && println("projection for |ϕ>=|", join(basis, ""), "> == ", res)
    return res
end

function projection_all(sys::MPS, message::String="")
    println(message)
    nqubits = length(sys.data)
    for x in 0:(2^nqubits - 1)
        n = reverse(digits(x, base=2))
        basis = [[0 for _ in 1:(nqubits-length(n))];n]
        projection(sys, basis, true)
    end
end

function measure(sys::MPS, i::Int64, output_probability::Bool=false)
    # restore collapsed state for |0> and |1>
    state_collapse₀ = permutedims(contract(sys.data[i], [1 0;0 0], (2,), (2,)), [1,3,2])
    state_collapse₁ = permutedims(contract(sys.data[i], [0 0;0 1], (2,), (2,)), [1,3,2])

    # assume collapse |0> then get that probability
    sys.data[i] = state_collapse₀
    nqubits = length(sys.data)
    t = contract(sys.data[nqubits], sys.data[nqubits], (2,), (2,))
    for x in (nqubits-1):-1:1
        t₀ = contract(sys.data[x], sys.data[x], (2,), (2,))
        t = contract(t, t₀, (1,3), (2,4))
        # these three methods should be contract to one method, but i don't know how to yet
        t = permutedims(t, (2,1,3,4))
        t = permutedims(t, (1,2,4,3))
        t = permutedims(t, (4,2,3,1))
    end

    probability₀ = reshape(t, 1)[1]
    if output_probability
        println("probability for |0> : ", probability₀)
    end

    # collapse #i qubit and return the classical information
    if rand() < probability₀
        sys.data[i] = state_collapse₀./sqrt(probability₀)
        return 0
    else
        sys.data[i] = state_collapse₁./sqrt(1-probability₀)
        return 1
    end
end

end # end for module
