using Random
using LinearAlgebra
using BenchmarkTools
using CUDA

function matmult(A::Matrix{Float64}, T::Int64)
    
    N = size(A)[1]
    B = rand(N, N)
    for i in 1:T
        B *= A 
    end 

    return B::Matrix{Float64}
end 

function matmultcu(A::Matrix{Float64}, T::Int64)
    
    A = CUDA.cu(A)
    N = size(A)[1]
    B = CUDA.cu(rand(N, N))
    for i in 1:T
        B *= A 
    end 
    
    B = B |> Matrix{Float64}    

    return B::Matrix{Float64}
end 

