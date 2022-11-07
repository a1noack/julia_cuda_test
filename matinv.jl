using Random
using LinearAlgebra
using BenchmarkTools
using CUDA

"""
    matinv(A::Matrix{Float64}, T::Int64}

Invert matrix `A` `T` times using CPU.
"""
function matinv(A::Matrix{Float64}, T::Int64)

    for i in 1:T
        inv(A)
    end
end 


"""
    matinvcuda(A::Matrix{Float64}, T::Int64}

Invert matrix `A` `T` times using GPU.
"""
function matinvcuda(A::Matrix{Float64}, T::Int64)

    A = CUDA.cu(copy(A))
    B = I(size(A)[1]) |> Matrix{Float64} |> CUDA.cu
    for i in 1:T
        A \ B
    end
end
