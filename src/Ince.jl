using LinearAlgebra     # linear algebra library

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
