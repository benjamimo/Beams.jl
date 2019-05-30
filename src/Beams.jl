module Beams

include("SpecFunctions.jl")    # Some Special functions
using SpecialFunctions

export LaguerreGaussBeam, SmallCoreBeam, HermiteGaussBeam, IGBeamE, IGBeamO, BesselGaussBeam, CosineGaussBeam, SineGaussBeam

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
    BesselGaussBeam(x, y, w0, phi, a, l)

Bessel-Gaussian beam """
function BesselGaussBeam(x::Float64, y::Float64, w0::Float64, phi::Float64, a::Float64, l::Int64)
    BG::ComplexF64 = 0.0 + im*0.0
    rr2 = (x^2 + y^2)/(w0^2)
    rr = sqrt(rr2)
    BG = exp(-rr2) * besselj(l, a*rr) * exp(im*(l*atan(y, x) + phi))
    return BG
end

"""
    CosineGaussBeam(x, y, w0, phi, a, l)

Cosine-Gaussian beam """
function CosineGaussBeam(x::Float64, y::Float64, w0::Float64, phi::Float64, a::Float64, th::Float64)
    CG::ComplexF64 = 0.0 + im*0.0
    rr2 = (x^2 + y^2)/(w0^2)
    CG = exp(-rr2) * cos(a*(x*cos(th) + y*sin(th))/w0)
    return CG
end

"""
    SineGaussBeam(x, y, w0, phi, a, l)

Sine-Gaussian beam """
function SineGaussBeam(x::Float64, y::Float64, w0::Float64, phi::Float64, a::Float64, th::Float64)
    SG::ComplexF64 = 0.0 + im*0.0
    rr2 = (x^2 + y^2)/(w0^2)
    SG = exp(-rr2) * sin(a*(x*cos(th) + y*sin(th))/w0)
    return SG
end


end # module
