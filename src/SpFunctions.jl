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
