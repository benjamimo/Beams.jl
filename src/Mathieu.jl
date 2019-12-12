using LinearAlgebra     # linear algebra library
using MathieuFunctions

# This code is based on the matrice method

function AMathieuCoef(r,q)

Ne = 25
if r%2 == 0

    # Matrix
    N = Ne+r
    dl = vcat(2*q, q*ones(N-1))
    d = 1.0*vcat(collect(0:2:2*N).^2)
    du = q*ones(N)
    M = Tridiagonal(dl,d,du)
    Mreg = convert(Array,M)

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    pos = div(r,2) + 1
    coef = A[:,index[pos]]

    # Normalization and coefficient calculation
    N1 = sqrt(coef[1]^2 + sum(coef.^2))
    NS = sign(coef[1])
    coef = coef/N1/NS

else

    # Matrix
    N = Ne+r
    dl = 1.0*vcat(q*ones(N))
    d = 1.0*vcat(1+q, collect(3:2:2*N+1).^2)
    du = 1.0*vcat(q*ones(N))
    M = Tridiagonal(dl,d,du)
    Mreg = convert(Array,M)

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    pos = div(r+1,2)
    coef = A[:,index[pos]]

    # Normalization and coefficient calculation
    N1 = sqrt(sum(coef.^2))
    NS = sign(coef[1])
    coef = coef/N1/NS
end
return A
end


function BMathieuCoef(r,q)

Ne = 25  # Size of the matrix, 50 is big enough
if r%2 == 0

    # Matrix
    N = Ne+r
    dl = q*ones(N)
    d = 1.0*vcat(collect(2:2:2*(N+1)).^2)
    du = q*ones(N)
    M = Tridiagonal(dl,d,du)
    Mreg = convert(Array,M)

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    pos = div(r,2)
    coef = A[:,index[pos]]

    # Normalization and coefficient calculation
    N1 = sqrt(sum(coef.^2))
    NS = sign(coef[1])
    coef = coef/N1/NS

else

    # Matrix
    N = Ne+r
    dl = 1.0*vcat(q*ones(N))
    d = 1.0*vcat(1-q, collect(3:2:2*N+1).^2)
    du = 1.0*vcat(q*ones(N))
    M = Tridiagonal(dl,d,du)
    Mreg = convert(Array,M)

    # Eigenvalues and Eigenvectors
    ets,A = eigen(Mreg)
    etss = sort(ets)
    index = sortperm(ets)
    pos = div(r+1,2)
    coef = A[:,index[pos]]

    # Normalization and coefficient calculation
    N1 = sqrt(sum(coef.^2))
    NS = sign(coef[1])
    coef = coef/N1/NS
end
return coef
end


function ceMathieu(r, q, coefs, z)
N = size(coefs,1)
if r%2 == 0
    k = 1.0*collect(0:N-1)
    ceP = cos.(2*k*z)'*coefs
else
    k = 2.0*collect(0:N-1) .+ 1.0
    ceP = cos.(k*z)'*coefs
end
return ceP
end


function seMathieu(r, q, coefs, z)
N = size(coefs,1)
if r%2 == 0
    k = 2.0*collect(0:N-1) .+ 2
    seP = sin.(k*z)'*coefs
else
    k = 2.0*collect(0:N-1) .+ 1.0
    seP = sin.(k*z)'*coefs
end
return seP
end


function JeMathieu(r, q, coefs, z)
N = size(coefs,1)
if r%2 == 0
    k = 1.0*collect(0:N-1)
    JeP = cosh.(2*k*z)'*coefs
else
    k = 2.0*collect(0:N-1) .+ 1.0
    JeP = cosh.(k*z)'*coefs
end
return JeP
end


function JoMathieu(r, q, coefs, z)
N = size(coefs,1)
if r%2 == 0
    k = 1.0*collect(1:N)
    JoP = sinh.(2*k*z)'*coefs
else
    k = 2.0*collect(0:N-1) .+ 1.0
    JoP = sinh.(k*z)'*coefs
end
return JoP
end
