
shape(a::AbstractArray) = size(a)

const Tensor{T, N} = Array{T, N}

function Base.kron(a::AbstractArray{Ta, N}, b::AbstractArray{Tb, N}) where {Ta, Tb, N}
    N == 0 && error("empty tensors.")
    sa = shape(a)
    sb = shape(b)
    sc = Tuple(sa[i]*sb[i] for i=1:N)
    c = Tensor{promote_type(Ta, Tb), N}(undef, sc)
    ranges = Vector{UnitRange{Int}}(undef, N)
    for index in CartesianIndices(a)
        # ranges[1] = (index[1]*sb[1]+1):(index[1]+1)*sb[1]
        for j = 1:N
            ranges[j] = ((index[j]-1)*sb[j]+1):(index[j]*sb[j])
        end
        c[ranges...] = a[index]*b
    end
    return c
end



function _group_extent(extent::NTuple{N, Int}, idx::NTuple{N1, Int}) where {N, N1}
    ext = Vector{Int}(undef, N1)
    l = 0
    for i=1:N1
        ext[i] = prod(extent[(l+1):(l+idx[i])])
        l += idx[i]
    end
    return NTuple{N1, Int}(ext)
end


function tie(a::AbstractArray{T, N}, axs::NTuple{N1, Int}) where {T, N, N1}
    (sum(axs) != N) && error("total number of axes should equal to tensor rank.")
    return reshape(a, _group_extent(shape(a), axs))
end



"""	
	move_selected_index_forward(a, I)
	move the indexes specified by I to the front of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_forward(a::Vector{T}, I) where {T}
    na = length(a)
    nI = length(I)
    b = Vector{T}(undef, na)
    k1 = 0
    k2 = nI
    for i=1:na
        s = 0
        while s != nI
        	if i == I[s+1]
        		b[s+1] = a[k1+1]
        	    k1 += 1
        	    break
        	end
        	s += 1
        end
        if s == nI
        	b[k2+1]=a[k1+1]
        	k1 += 1
            k2 += 1
        end
    end
    return b
end

function move_selected_index_forward(a::NTuple{N, T}, I) where {N, T}
    return NTuple{N, T}(move_selected_index_forward([a...], I))
end

"""	
	move_selected_index_backward(a, I)
	move the indexes specified by I to the back of a
	# Arguments
	@ a::NTuple{N, Int}: the input tensor.
	@ I: tuple or vector of integer.
"""
function move_selected_index_backward(a::Vector{T}, I) where {T}
	na = length(a)
	nI = length(I)
	nr = na - nI
	b = Vector{T}(undef, na)
	k1 = 0
	k2 = 0
	for i = 1:na
	    s = 0
	    while s != nI
	    	if i == I[s+1]
	    		b[nr+s+1] = a[k1+1]
	    		k1 += 1
	    		break
	    	end
	    	s += 1
	    end
	    if s == nI
	        b[k2+1] = a[k1+1]
	        k2 += 1
	        k1 += 1
	    end
	end
	return b
end

function move_selected_index_backward(a::NTuple{N, T}, I) where {N, T}
	return NTuple{N, T}(move_selected_index_backward([a...], I))
end

function contract(a::AbstractArray{Ta, Na}, b::AbstractArray{Tb, Nb}, axs::Tuple{NTuple{N, Int}, NTuple{N, Int}}) where {Ta, Na, Tb, Nb, N}
    ia, ib = axs
    seqindex_a = move_selected_index_backward(collect(1:Na), ia)
    seqindex_b = move_selected_index_forward(collect(1:Nb), ib)
    ap = permutedims(a, seqindex_a)
    bp = permutedims(b, seqindex_b)
    return reshape(tie(ap, (Na-N, N)) * tie(bp, (N, Nb-N)), shape(ap)[1:(Na-N)]..., shape(bp)[(N+1):Nb]...)
end

