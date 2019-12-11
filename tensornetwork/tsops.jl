module MyTensorOps
export ⨁, ⨂, contract!

# direct sum
function ⨁(𝑨::AbstractArray{𝕋, ℕ}, 𝑩::AbstractArray{𝕋, ℕ}, axes::Tuple) where {𝕋, ℕ}
    𝑪 = []

    s𝑨 = size(𝑨) # store size of 𝑨 and 𝑩 to calculate size of 𝑪
    s𝑩 = size(𝑩)
    s𝑪 = []

    if length(axes) == ℕ
        s𝑪 = [x+y for (x, y) in zip(s𝑨, s𝑩)]
        𝑪 = zeros(s𝑪...)
        𝑩start = CartesianIndex([x+1 for x in s𝑨]...)
        𝑩end = CartesianIndex(s𝑪...)
    else
        if sum([[x!=y for (x, y) in zip(s𝑨, s𝑩)][x] for x in axes]) != 0
            error("parameter wrong")
        end

        # if x in axes, this dim's data will not expand
        #for x in 1:N if x in axes push!(s𝑪, s𝑨[x]) else push!(s𝑪, s𝑨[x]+s𝑩[x]) end end
        [x in axes ? s𝑪 : s𝑨[x]+s𝑩[x] for x in 1:N] # NOTES:P2
        𝑪 = zeros(s𝑪...)

        # construct 𝑩start
        tmp = ones(Int64, N) # NOTES:P1
        for x in 1:N if !(x in axes) tmp[x]=1+s𝑨[x] end end
        𝑩start = CartesianIndex((tmp...,))
        𝑩end = CartesianIndex(s𝑪...)
    end

    𝑨start = CartesianIndices(𝑪)[1]
    𝑨end = CartesianIndex(s𝑨...)
    𝑪[𝑨start:𝑨end] = 𝑨
    𝑪[𝑩start:𝑩end] = 𝑩

    return 𝑪
end

function contract!(a::AbstractArray, b::AbstractArray, dima::Tuple, dimb::Tuple, debug=false)
    debug && println(size(a), "+++", size(b), "---", dima, "---", dimb)
    if [size(a)[x] for x in dima] != [size(b)[x] for x in dimb] error("size is wrong") end
    if length(dima) == 0
        a_size = [size(a)...]
        b_size = [size(b)...]
        ra = reshape(a, prod(a_size), 1)
        rb = reshape(b, 1, prod(b_size))
        return reshape(ra*rb, Tuple([a_size; b_size]))
    end

    a_left = filter(x -> !(x in dima), [x for x in 1:ndims(a)])
    a_right = collect(dima)
    a_perm = [a_left; a_right]
    a_reshape_size = [size(a)[x] for x in a_left]
    a_left_len = prod(a_reshape_size)
    a_right_len = prod([size(a)[x] for x in a_right])

    b_left = collect(dimb)
    b_right = filter(x -> !(x in dimb), [x for x in 1:ndims(b)])
    b_perm = [b_left; b_right]
    b_left_len = prod([size(b)[x] for x in b_left])
    b_reshape_size = [size(b)[x] for x in b_right]
    b_right_len = prod(b_reshape_size)

    am = permutedims(a, a_perm)
    bm = permutedims(b, b_perm)

    ra = reshape(a, (a_left_len, a_right_len))
    rb = reshape(b, (b_left_len, b_right_len))

    return reshape(ra*rb, Tuple([a_reshape_size; b_reshape_size]))
end

# The right version for kron
function ⨂(t1::AbstractArray, t2::AbstractArray)
    if ndims(t1) != ndims(t2)
        error("ndims t1 must be equaled to t2")
    end

    T = promote_type(eltype(t2), eltype(t1))
    ret = zeros(T, size(t1).*size(t2))
    for i in CartesianIndices(t1)
        c_start = CartesianIndex(.+(.*(.-(Tuple(i), 1), size(t2)), 1))
        c_end = CartesianIndex(Tuple(i).*size(t2))
        ret[c_start:c_end] = t1[i] * t2
    end
    return ret
end

end
