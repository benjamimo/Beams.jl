module Beams

include("SpecFunctions.jl")    # Some Special functions
using SpecialFunctions   # needed for Bessel!
using GSL  # Mathieu functions!!

export LaguerreGaussBeam, SmallCoreBeam, HermiteGaussBeam, IGBeamE, IGBeamO,
       BesselBeam, CosineBeam, SineBeam, MathieuBeamE, MathieuBeamO, ParabolicBeamE,
       ParabolicBeamO, FAiryBeam, GaussianBeam, HzGprop

"""
    LaguerreGaussBeam(x, y, w0, phi, l, p)

Laguerre-Gaussian beam """
#function LaguerreGaussBeam(x::Float64, y::Float64, w0::Float64, phi::Float64, l::Int64, p::Int64)
function LaguerreGaussBeam(x::Float64, y::Float64, w0::Float64, phi::Float64, l::Int64, p::Int64)
    LG::ComplexF64=0.0 + im*0.0
    rr2=(x^2 + y^2)/(w0^2)

    if p==0
        LG=(sqrt(rr2)^abs(l)) * exp(-rr2) * exp(im*(l*atan(y,x)+phi))
        # println(p)
    else
        LG=(sqrt(rr2)^abs(l)) * exp(-rr2) * LaguerrePoly.(p, abs(l), 2*rr2) * exp(im*(l*atan(y,x)+phi))   # Im not sure about adding the dot in LaguerrePoly...
        # println(p)
    end

    return LG::ComplexF64
end

"""
    SmallCore(x, y, w0, wV, phi, l)

SmallCore vortex beam """
function SmallCoreBeam(x::Float64, y::Float64, w0::Float64, wV::Float64, phi::Float64, l::Float64)
    SC::ComplexF64=0.0 + im*0.0
    SC= tanh((sqrt(x^2+y^2)) / wV) * exp((-x^2 - y^2)/(w0^2)) * exp(im*(l*atan(y, x) + phi))
    #SC= tanh((sqrt(x^2+y^2)) / wV) * exp((-x^2 - y^2)/(w0^2)) * cis(l*atan(y, x) + phi)
    return SC::ComplexF64
end

"""
    HermiteGaussBeam(x, y, w0, phi, l, p)

Hermite-Gaussian beam """
function HermiteGaussBeam(x::Float64, y::Float64, w0::Float64, phi::Float64, m::Int64, n::Int64)
    HG::ComplexF64=0.0 + im*0.0
    rr2=(x^2 + y^2)/(w0^2)
    HG= HermitePoly(m,sqrt(2)*x/w0) * HermitePoly(n,sqrt(2)*y/w0) * exp(-rr2) * exp(im*(phi))   # Im not sure about adding the dot in LaguerrePoly...
    return HG::Complex{Float64}
end

"""
    IGBeamE(x, y, w0, p, m, q)

Even Ince-Gaussian beam """
function IGBeamE(x::Float64, y::Float64, w0::Float64, phi::Float64, p::Int64, m::Int64, q::Float64, Coef::Array{Float64,1})
    IG::ComplexF64 = 0.0 + im*0.0
    ff = sqrt(q/2) * w0
    rr2 = (x^2 + y^2)/(w0^2)
    #Coef = CInceCoef(p,m,q)
    uu = acosh((x+im*y)/ff)
    ee = real(uu)
    nn = imag(uu)
    nn = nn + (nn<0)*2*pi
    IG = CInce(p,m,q,Coef,nn) * CInce(p,m,q,Coef,im*ee) * exp(-rr2) * exp(im*phi)
    return IG::Complex{Float64}
end

"""
    IGBeamO(x, y, w0, p, m, q)

Even Ince-Gaussian beam """
function IGBeamO(x::Float64, y::Float64, w0::Float64, phi::Float64, p::Int64, m::Int64, q::Float64, Coef::Array{Float64,1})
    IG::ComplexF64 = 0.0 + im*0.0
    ff = sqrt(q/2) * w0
    rr2 = (x^2 + y^2)/(w0^2)
    uu = acosh((x+im*y)/ff)
    ee = real(uu)
    nn = imag(uu)
    nn = nn + (nn<0)*2*pi
    IG = SInce(p,m,q,Coef,nn) * SInce(p,m,q,Coef,im*ee) * exp(-rr2) * exp(im*phi)
    return IG::Complex{Float64}
end

"""
    BesselGaussBeam(x, y, muz, phi, kt, l)

Bessel-Gaussian beam """
function BesselBeam(x::Float64, y::Float64, muz::T, phi::Float64, kt::Float64, l::Int64) where T <: Union{Float64, ComplexF64}
    BG::ComplexF64 = 0.0 + im*0.0
    rr = sqrt(x^2 + y^2)
    BG = besselj(l, kt*rr/muz) * exp(im*(l*atan(y, x) + phi))
    return BG
end

"""
    CosineGaussBeam(x, y, phi, kt, th)

Cosine-Gaussian beam """
function CosineBeam(x::Float64, y::Float64, muz::T, phi::Float64, kt::Float64, th::Float64) where T <: Union{Float64, ComplexF64}
    CG::ComplexF64 = 0.0 + im*0.0
    CG = cos(kt*(x*cos(th) + y*sin(th))/muz)
    return CG
end

"""
    SineGaussBeam(x, y, muz, phi, kt, th)

Sine-Gaussian beam """
function SineBeam(x::Float64, y::Float64, muz::T, phi::Float64, kt::Float64, th::Float64) where T <: Union{Float64, ComplexF64}
    SG::ComplexF64 = 0.0 + im*0.0
    SG = sin(kt*(x*cos(th) + y*sin(th))/muz)
    return SG
end

"""
    MathieuGaussBeamE(x, y, phi, m, q, kt)

Mathieu-Gaussian beam (even)"""
function MathieuBeamE(x::Float64, y::Float64, phi::Float64, m::Int64, q::Float64, kt::Float64) where T <: Union{Float64, ComplexF64}
    MGE::ComplexF64 = 0.0 + im*0.0
    f0 = 2*sqrt(q)/kt
    uu = acosh((x+im*y)/f0)
    ee = real(uu)
    nn = imag(uu)
    nn = nn + (nn<0)*2*pi
    MGE = sf_mathieu_Mc(1,m,q,ee) * sf_mathieu_ce(m,q,nn) * exp(im*phi)
    return MGE
end

"""
    MathieuGaussBeamO(x, y, phi, m, q, kt)

Mathieu-Gaussian beam (odd)"""
function MathieuBeamO(x::Float64, y::Float64, phi::Float64, m::Int64, q::Float64, kt::Float64) where T <: Union{Float64, ComplexF64}
    MGO::ComplexF64 = 0.0 + im*0.0
    f0 = 2*sqrt(q)/kt
    uu = acosh((x+im*y)/f0)
    ee = real(uu)
    nn = imag(uu)
    nn = nn + (nn<0)*2*pi
    MGO = sf_mathieu_Ms(1,m,q,ee) * sf_mathieu_se(m,q,nn) * exp(im*phi)
    return MGO
end

"""
    ParabolicGaussBeamE(x, y, phi, a, kt, gamma1)

Parabolic-Gaussian beam (even)"""
function ParabolicBeamE(x::Float64, y::Float64, muz::T, phi::Float64, a::Float64, kt::Float64, g1::Float64) where T <: Union{Float64, ComplexF64}
    PGE::ComplexF64 = 0.0 + im*0.0
    uu = (2*(x + im*y))^(1/2)
    nn = real(uu)
    ee = imag(uu)
    PGE = g1 * ParabolicEven(sqrt(2*kt/muz)*ee,a) * ParabolicEven(sqrt(2*kt/muz)*nn,-a)
    return PGE
end


"""
    ParabolicGaussBeamO(x, y, phi, a, kt, gamma3)

Parabolic-Gaussian beam (odd)"""
function ParabolicBeamO(x::Float64, y::Float64, muz::T, phi::Float64, a::Float64, kt::Float64, g3::Float64) where T <: Union{Float64, ComplexF64}
    PGO::ComplexF64 = 0.0 + im*0.0
    uu = (2*(x + im*y))^(1/2)
    nn = real(uu)
    ee = imag(uu)
    PGO = g3 * ParabolicOdd(sqrt(2*kt/muz)*ee,a) * ParabolicOdd(sqrt(2*kt/muz)*nn,-a)
    return PGO
end

"""
    FAiryBeam(kx, ky, x0, a)

Fourier Spectrum of Airy beam"""
function FAiryBeam(kx::Float64, ky::Float64, x0::Float64, a::Float64)
    FAB::ComplexF64 = 0.0 + im*0.0
    kk2 = (kx^2 + ky^2)
    FAB = exp(im*(x0^3)*((kx^3)/3 + (ky^3)/3)) * exp(-a*(x0^2)*kk2)
    return FAB
end

"""
    GaussianBeam(x, y, z, w0, k)

Fundamental Gaussian beam """
function GaussianBeam(x::Float64, y::Float64, z, w0, k)
    GB::ComplexF64 = 0.0 + im*0.0
    zr = k*(w0^2)/2
    muz = 1 + im*z/zr
    rr2m = (x^2 + y^2)/(muz*w0^2)
    GB = ((exp(im*k*z))/muz) * exp(-rr2m)
    return GB
end

"""
    HzGprop(z, w0, k, kt)

Propagation term of HzG beams """
function HzGprop(z::Float64, w0::Float64, k::Float64, kt::Float64)
    HGP::ComplexF64 = 0.0 + im*0.0
    zr = k*(w0^2)/2
    muz = 1 + im*z/zr
    HGP = exp(-im*(kt^2)*z/(2*k*muz))
    return HGP
end

end # module
