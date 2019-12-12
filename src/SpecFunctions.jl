using LinearAlgebra     # linear algebra library
using PyCall

"""
    LaguerrePoly(n,k,x)

Compute the Associated Laguerre Polynomial """
function LaguerrePoly(n::Int64,k::Int64,x::Float64)
    term1=1/factorial(n)
    L=0
    for ii=0:n
        L=L+(factorial(n)/factorial(ii)) * binomial(k+n,n-ii) * (-x)^ii
    end
    L=term1 * L
end

"""
    HermitePoly(n,x)

Compute the Hermite Polynomial """
function HermitePoly(n::Int64, x::Float64)
    H=0
    floorn=div(n,2)
    for k=0:floorn
        H=H+(((-1)^k) * (2^(-2*k+n)) * (x^(-2*k+n)))/(factorial(k) * factorial(-2*k+n))
    end
    H=factorial(n) * H
end


function CInceCoef(p::Int64,m::Int64,q::Float64)
if p%2 == 0
    # Parameters
    j=div(p,2)
    N=j+1
    n=div(m,2)+1

    # Matrix
    dl=vcat(2*q*j, q*(j.-(1:N-2)))
    d=vcat(0,4.0*((0:N-2).+1).^2)
    du=q.*(j.+collect(1:N-1))
    M=Tridiagonal(dl,d,du)
    Mreg=convert(Array,M)

    if p==0
        M=0
    end

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    A=A[:,index]

    # Normalization
    N2=2*A[1,n].^2+sum(A[2:N,n].^2)
    NS=sign(sum(A[:,n]))
    A=A/sqrt(N2)*NS

else
    # Parameters
    j=div((p-1),2)
    N=j+1
    n=div((m+1),2)

    # Matrix
    dl=q/2*(p.-(2*collect(1:N-1).-1))
    d=vcat(q/2 .+ p*q/2 .+ 1, 1.0*(2*(1:N-1) .+ 1).^2)
    du=q/2*(p.+(2*collect(0:N-2).+3))
    M=Tridiagonal(dl,d,du)
    Mreg=convert(Array,M)

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    A=A[:,index]

    # Normalization
    N2=sum(A[:,n].^2)
    NS=sign(sum(A[:,n]))
    A=A/sqrt(N2)*NS
end

return A[:,n]
end


function SInceCoef(p::Int64,m::Int64,q::Float64)
if p%2 == 0
    # Parameters
    j=div(p,2)
    N=j+1
    n=div(m,2)

    # Matrix
    dl=q*(j .- collect(1:N-2))
    d=4.0*(collect(0:N-2) .+ 1).^2
    du=q.*(j .+ collect(2:N-1))
    M=Tridiagonal(dl,d,du)
    Mreg=convert(Array,M)

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    A=A[:,index]

    # Normalization
    r=collect(1:N-1)
    N2=sum(A[:,n].^2)
    NS=sign(sum(r .* A[:,n]))
    A=A/sqrt(N2)*NS

else
    # Parameters
    j=div((p-1),2)
    N=j+1
    n=div((m+1),2)

    # Matrix
    dl=q/2*(p .- (2*collect(1:N-1) .- 1))
    d=vcat(-q/2-p*q/2+1, (2.0*(1:N-1) .+ 1).^2)
    du=q/2*(p .+ (2*collect(0:N-2) .+ 3))
    M=Tridiagonal(dl,d,du)
    Mreg=convert(Array,M)

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    A=A[:,index]

    # Normalization
    r=2*collect(0:N-1) .+ 1
    N2=sum(A[:,n].^2)
    NS=sign(sum(r .* A[:,n]))
    A=A/sqrt(N2)*NS
end
return A[:,n]
end

function CInce(p::Int64,m::Int64,q::Float64,InceCoef::Array{Float64,1},z::T) where T <: Union{Float64, ComplexF64}
## Preallocate

## Calculate the Coefficients
if p%2 == 0
    #### p Even ####
    #m=4; p=6; q=0.5; z=1 # Examples

    # Parameters
    j=div(p,2)
    N=j+1
    n=div(m,2)+1

    # Ince Polynomial
    r=collect(0:N-1)
    #IP=cos.(2*z*r)'*A[:,n]
    IP=cos.(2*z*r)'*InceCoef
    #eta=etss[n]

else
    #### p ODD ###
    #p=5; m=3; q=0.5; z=1 # Examples

    # Parameters
    j=div((p-1),2)
    N=j+1
    n=div((m+1),2)

    # Ince Polynomial
    r=2*collect(0:N-1) .+ 1;
    #IP=cos.(z*r)'*A[:,n]
    IP=cos.(z*r)'*InceCoef
    #eta=etss[n]
end

return IP
end


function SInce(p::Int64,m::Int64,q::Float64,InceCoef::Array{Float64,1},z::T) where T <: Union{Float64, ComplexF64}
## Preallocate

## Calculate the Coefficients
if p%2 == 0
    #### p Even ####
    #m=4; p=6; q=0.5; z=1 # Examples

    # Parameters
    j=div(p,2)
    N=j+1
    n=div(m,2)

    # Ince Polynomial
    r=collect(1:N-1)
    #IP=sin.(2*z*r)'*A[:,n]
    IP=sin.(2*z*r)'*InceCoef
    #eta=etss[n]

else
    #### p ODD ###
    #p=5; m=3; q=0.5; z=1 # Examples

    # Parameters
    j=div((p-1),2)
    N=j+1
    n=div((m+1),2)

    # Ince Polynomial
    r=2*collect(0:N-1) .+ 1
    #IP=sin.(z*r)'*A[:,n]
    IP=sin.(z*r)'*InceCoef
    #eta=etss[n]
end

return IP
end


function ParabolicEven(x,a)
    mpm = pyimport("mpmath")  # hypergeometric confluent function!
    pe = mpm.hyp1f1(0.25-im*0.5*a, 0.5, im*0.5*x^2)
    pej = convert(Complex, pe)
    PE = exp(-0.25*im*x^2) * pej
    return PE
end


function ParabolicOdd(x,a)
    mpm = pyimport("mpmath")  # hypergeometric confluent function!
    po = mpm.hyp1f1(0.75-im*0.5*a, 1.5, im*0.5*x^2)
    poj = convert(Complex, po)
    PO = x*exp(-0.25*im*x^2) * poj
    return PO
end
