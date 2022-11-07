using Random
using LinearAlgebra
using BenchmarkTools
using CUDA

"""
    matmult(A::Matrix{Float64}, T::Int64)

Multiply matrix `A` by itself `T` times using CPU.
"""
function matmult(A::Matrix{Float64}, T::Int64)
    
    for i in 1:T
        A *= A 
    end 

    return A::Matrix{Float64}
end 

"""
    matmultcuda(A::Matrix{Float64}, T::Int64)

Multiply matrix `A` by itself `T` times using GPU.
"""
function matmultcuda(A::Matrix{Float64}, T::Int64)
    
    # Convert `A` to CUDA matrix type
    A = CUDA.cu(A)
    for i in 1:T
        A *= A 
    end 
    # Convert `A` back to regular matrix w/ Float64 entries 
    A = A |> Matrix{Float64}    

    return A::Matrix{Float64}
end 

