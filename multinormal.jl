using Random
using LinearAlgebra
using CUDA
using BenchmarkTools

"""
    multinormal(mu::Vector{Float64}, Sigma::Matrix{Float64})

Draw vector from multivariate normal distribution N(mu, Sigma) using CPU.
"""
function multinormal(μ::Vector{Float64}, Σ::Matrix{Float64})

    nvar = size(Σ)[1]         # Num. of variables 

    if (0 in diag(Σ)) == false  # No degenerate random vars.
        Q = cholesky(Hermitian(Σ), Val(true), check=false).U       # Upper triang. Cholesky mat.  
        X = Q * randn(length(μ)) + μ    # Multiv. normal vector draw  
    else                        # in case of degenerate random vars.
        keep = Any[]
        for i in 1:nvar
            if Σ[i, i] != 0
                push!(keep, i)
            end
        end
        Σsub = Σ[keep, keep]
        μsub = μ[keep]
        Q = cholesky(Hermitian(Σsub), Val(true), check=false).U       # Upper triang. Cholesky mat.  
        Xsub = Q * randn(length(μsub)) + μsub    # Multiv. normal vector draw  
        X = zeros(nvar)
        j = 1
        for i in 1:nvar
            if i in keep    # If i-th var. is non-degen. 
                X[i] = Xsub[j]
                j = j + 1
            else
                X[i] = μ[i] # If i-th var. is degen. 
            end
        end
    end

    return X::Vector{Float64}
end

"""
    multinormalcuda(mu::Vector{Float64}, Sigma::Matrix{Float64})

Draw vector from multivariate normal distribution N(mu, Sigma) using GPU.
"""
function multinormalcuda(μ::Vector{Float64}, Σ::Matrix{Float64})

    nvar = size(Σ)[1]         # Num. of variables 

    if (0 in diag(Σ)) == false  # No degenerate random vars.
        Q = CUDA.cu(cholesky(Hermitian(Σ), Val(true), check=false).U)       # Upper triang. Cholesky mat.  
        X = Q * CUDA.randn(length(μ)) + CUDA.cu(μ)    # Multiv. normal vector draw  
    else                        # in case of degenerate random vars.
        keep = Any[]
        for i in 1:nvar
            if Σ[i, i] != 0
                push!(keep, i)
            end
        end
        Σsub = Σ[keep, keep]
        μsub = μ[keep]
        Q = CUDA.cu(cholesky(Hermitian(Σsub), Val(true), check=false).U)       # Upper triang. Cholesky mat.  
        Xsub = Q * CUDA.randn(length(μsub)) + CUDA.cu(μsub)    # Multiv. normal vector draw  
        X = zeros(nvar)
        j = 1
        for i in 1:nvar
            if i in keep    # If i-th var. is non-degen. 
                X[i] = Xsub[j]
                j = j + 1
            else
                X[i] = μ[i] # If i-th var. is degen. 
            end
        end
    end
    X = X |> Vector{Float64}

    return X::Vector{Float64}
end
