#include "CausalRenewalDevice.h"
#include "macrofacet/mathutility/CausalMajorant.h"
#include <nanovdb/NanoVDB.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace mf::cuda_renewal {
namespace {
constexpr int threads=128, lanes=8, groups=threads/lanes;
__device__ void syncGroup() {
    const unsigned groupMask=0xffu << ((threadIdx.x%32)/lanes*lanes);
    __syncwarp(groupMask);
}
__host__ __device__ V3 operator+(V3 a,V3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__host__ __device__ V3 operator-(V3 a,V3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__host__ __device__ V3 operator*(V3 a,double b) { return {a.x*b,a.y*b,a.z*b}; }
__device__ V3 product(V3 a,V3 b) { return {a.x*b.x,a.y*b.y,a.z*b.z}; }
__device__ double dot(V3 a,V3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ V3 unit(V3 a) { return a*(1/sqrt(dot(a,a))); }
__device__ V3 cross(V3 a,V3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
__device__ double component(V3 a,int j) { return j==0 ? a.x : j==1 ? a.y : a.z; }
__device__ double clamp(double a,double lo,double hi) { return fmin(hi,fmax(lo,a)); }
__device__ V3 clip(V3 a,const Scene& c) {
    return {clamp(a.x,c.minimum.x,c.maximum.x),clamp(a.y,c.minimum.y,c.maximum.y),clamp(a.z,c.minimum.z,c.maximum.z)};
}
__device__ void frame(V3 w,V3& u,V3& v) {
    w=unit(w); u=unit(cross(fabs(w.z)<.9 ? V3{0,0,1} : V3{1,0,0},w)); v=cross(w,u);
}
__device__ bool intersect(V3 o,V3 d,const Scene& c,double& lo,double& hi) {
    lo=0; hi=CUDART_INF;
    for (int a=0;a<3;++a) {
        const double x=component(o,a),v=component(d,a),mn=component(c.minimum,a),mx=component(c.maximum,a);
        if (v==0) { if (x<mn || x>mx) return false; continue; }
        const double t0=(mn-x)/v,t1=(mx-x)/v;
        lo=fmax(lo,fmin(t0,t1)); hi=fmin(hi,fmax(t0,t1));
        if (hi<lo) return false;
    }
    return hi>lo;
}
struct Device {
    const nanovdb::NanoGrid<float>* grid;
    const float* weights;
    Network net;
    Scene scene;
    V3 gridOrigin;
    double dx;
};
struct Cell {
    double lo,hi,a,b,c,d;
    __device__ double value(double q) const { const double t=(q-lo)/(hi-lo); return ((a*t+b)*t+c)*t+d; }
    __device__ double derivative(double q) const { const double t=(q-lo)/(hi-lo); return ((3*a*t+2*b)*t+c)/(hi-lo); }
};
struct Flight {
    V3 origin, direction, birthGradient, birthMean, gradient;
    double maximum, remaining, hazard, q, speed, uniforms[3], normals[2];
    unsigned long long intervals, candidates, nulls, bounds;
    int surface, queryOnly, status, error, cellIndex;
    // status: 0 active, 1 hit, 2 escape, 3 invalid. No failure becomes a black sample.
};
struct Slot {
    Flight flight;
    float hidden[32];
    curandStatePhilox4_32_10_t rng;
    V3 throughput;
    unsigned long long path, totalIntervals, flights;
    int depth, active, needsBirth;
};
template<int H> struct Work {
    float x[64], y[64], a[3*H], b[3*H], h[H];
    double quad[8], left, right, mass, factor, lo, hi, residual, cursor, limit;
    Cell cell;
    Flight f;
};
__device__ double uniform(curandStatePhilox4_32_10_t& rng);
__device__ V3 gradientAt(V3 p,V3 interior,const Device& d,int& error) {
    const V3 u{(p.x-d.gridOrigin.x)/d.dx,(p.y-d.gridOrigin.y)/d.dx,(p.z-d.gridOrigin.z)/d.dx};
    const int ix=int(floor((interior.x-d.gridOrigin.x)/d.dx));
    const int iy=int(floor((interior.y-d.gridOrigin.y)/d.dx));
    const int iz=int(floor((interior.z-d.gridOrigin.z)/d.dx));
    const double x=u.x-ix,y=u.y-iy,z=u.z-iz;
    auto accessor=d.grid->getAccessor(); V3 g{0,0,0};
    for (int k=0;k<2;++k) for (int j=0;j<2;++j) for (int i=0;i<2;++i) {
        const nanovdb::Coord node(ix+i,iy+j,iz+k);
        const double v=accessor.getValue(node);
        if (!accessor.isActive(node) || !isfinite(v)) error=2;
        g.x+=(i?1:-1)*(j?y:1-y)*(k?z:1-z)*v;
        g.y+=(i?x:1-x)*(j?1:-1)*(k?z:1-z)*v;
        g.z+=(i?x:1-x)*(j?y:1-y)*(k?1:-1)*v;
    }
    return g*(1/d.dx);
}
// Recover exactly the same cellwise cubic as RayMeanProfile::fromField, on
// demand. Strict/right selection distinguishes network endpoint features from
// the open integration interval after a voxel face. No epsilon ray offset.
__device__ Cell cellAt(const Flight& f,double q,bool right,const Device& d) {
    double lo=0,hi=f.maximum;
    const double distance=q*d.scene.ell;
    for (int axis=0;axis<3;++axis) {
        const double v=component(f.direction,axis),o=component(f.origin,axis),g=component(d.gridOrigin,axis);
        if (fabs(v)<1e-14) continue;
        const int base=int(floor((o+distance*v-g)/d.dx));
        for (int j=-1;j<=2;++j) {
            const double age=(g+double(base+j)*d.dx-o)/v;
            const double t=age/d.scene.ell;
            if (!(t>0 && t<f.maximum)) continue;
            if (right ? t<=q : t<q) lo=fmax(lo,t); else hi=fmin(hi,t);
        }
    }
    const int pieces=max(1,int(ceil((hi-lo)/d.scene.profileStep)));
    if (pieces>1) {
        const double width=hi-lo;
        const double coordinate=(q-lo)/width*pieces;
        const int part=min(pieces-1,max(0,right ? int(floor(coordinate)) : int(ceil(coordinate))-1));
        const double begin=lo;
        lo=part==0 ? begin : begin+width*(double(part)/pieces);
        hi=part+1==pieces ? hi : begin+width*(double(part+1)/pieces);
    }
    const double width=hi-lo;
    double v[4],corners[8];
    const V3 interior=f.origin+f.direction*(d.scene.ell*(lo+hi)/2);
    const int ix=int(floor((interior.x-d.gridOrigin.x)/d.dx));
    const int iy=int(floor((interior.y-d.gridOrigin.y)/d.dx));
    const int iz=int(floor((interior.z-d.gridOrigin.z)/d.dx));
    auto accessor=d.grid->getAccessor();
    for (int k=0;k<2;++k) for (int j=0;j<2;++j) for (int i=0;i<2;++i) {
        const nanovdb::Coord node(ix+i,iy+j,iz+k);
        corners[i+2*j+4*k]=accessor.isActive(node) ? double(accessor.getValue(node)) : CUDART_NAN;
    }
    const double points[4]={lo,lo+width/3,lo+2*width/3,hi};
    for (int t=0;t<4;++t) {
        const V3 p=f.origin+f.direction*(d.scene.ell*points[t]);
        const double x=(p.x-d.gridOrigin.x)/d.dx-ix,y=(p.y-d.gridOrigin.y)/d.dx-iy,z=(p.z-d.gridOrigin.z)/d.dx-iz;
        double sum=0;
        for (int k=0;k<2;++k) for (int j=0;j<2;++j) for (int i=0;i<2;++i)
            sum+=(i?x:1-x)*(j?y:1-y)*(k?z:1-z)*corners[i+2*j+4*k];
        v[t]=sum/d.scene.sigma;
    }
    return {lo,hi,(-9*v[0]+27*v[1]-27*v[2]+9*v[3])/2,
        (18*v[0]-45*v[1]+36*v[2]-9*v[3])/2,(-11*v[0]+18*v[1]-9*v[2]+2*v[3])/2,v[0]};
}
__device__ double logCdf(double z) {
    constexpr double root2=1.4142135623730950488;
    if (z>0) return log1p(-.5*erfc(z/root2));
    const double p=.5*erfc(-z/root2);
    if (p>0) return log(p);
    const double x=-z; double term=1,sum=1,previous=CUDART_INF;
    for (int n=1;n<256;++n) {
        term*=-double(2*n-1)/(x*x);
        if (fabs(term)>=previous) break;
        sum+=term; previous=fabs(term);
        if (fabs(term)<1e-30*fabs(sum)) break;
    }
    return -.5*x*x-.91893853320467274178-log(x)+log(sum);
}
__device__ double rate(double b,double v) {
    // Algebraically identical positive-side Rice hazard. erfcx avoids both
    // overflow and repeated log/exp conversions in every quadrature node.
    constexpr double invRoot2=.70710678118654752440,invRoot2Pi=.39894228040143267794;
    const double mills=b<=0 ? .79788456080286535588/erfcx(-b*invRoot2)
        : invRoot2Pi*exp(-.5*b*b)/(1-.5*erfc(b*invRoot2));
    const double phi=invRoot2Pi*exp(-.5*v*v); double negativeMean;
    if (v<=0) negativeMean=phi-v*.5*erfc(v*invRoot2);
    else if (v<=26) negativeMean=phi*(1-v*1.2533141373155002512*erfcx(v*invRoot2));
    else {
        const double x=1/(v*v);
        negativeMean=phi*x*(1+x*(-3+x*(15+x*(-105+x*(945+x*(-10395+x*135135))))));
    }
    return mills*negativeMean;
}
__device__ double quantileLog(double p) {
    const bool reflect=p>-.69314718055994530942;
    if (reflect) p=log(-expm1(p));
    double lo=-2,hi=0;
    while (logCdf(lo)>p && isfinite(lo)) lo*=2;
    for (int i=0;i<100;++i) {
        const double mid=(lo+hi)/2;
        if (logCdf(mid)<p) lo=mid; else hi=mid;
        if (hi-lo<1e-14+1e-13*fabs(mid)) break;
    }
    const double x=(lo+hi)/2; return reflect ? -x : x;
}
__device__ float silu(float x) { return x/(1.f+expf(-x)); }
// Input-major weights make neighboring output neurons coalesced. An 8-lane group owns
// the recurrent state; barriers never cross a ray or depend on another ray.
__device__ void dense(Dense m,const float* input,float* output,const Device& d,int lane,bool activate=false) {
    for (int j=lane;j<m.output;j+=lanes) {
        float sum=0;
        for (int i=0;i<m.input;++i) sum+=d.weights[m.offset+i*m.output+j]*input[i];
        sum+=d.weights[m.bias+j]; output[j]=activate ? silu(sum) : sum;
    }
    syncGroup();
}
template<int H> __device__ void integrate(Work<H>& w,double lo,double hi,int lane) {
    constexpr double nodes[4]={.1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
    constexpr double weights[4]={.3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};
    if (lane<8) {
        const double q=(lo+hi)/2+(lane%2?1:-1)*(hi-lo)/2*nodes[lane/2];
        w.quad[lane]=weights[lane/2]*rate(w.cell.value(q),w.cell.derivative(q));
    }
    syncGroup();
    if (lane==0) { double sum=0; for (int i=0;i<8;++i) sum+=w.quad[i]; w.mass=(hi-lo)/2*sum; }
    syncGroup();
}
template<int H> __device__ void initialize(Work<H>& w,const Device& d,int lane) {
    if (lane==0) {
        auto& f=w.f;
        double lo,hi; f.direction=unit(f.direction);
        f.status=0; f.error=0; f.intervals=0; f.cellIndex=0; f.hazard=0; f.q=0; f.speed=0; f.gradient={0,0,0};
        f.candidates=f.nulls=f.bounds=0;
        if (d.scene.deltaTracking && !f.queryOnly) f.hazard=CUDART_NAN;
        if (!intersect(f.origin,f.direction,d.scene,lo,hi)) f.status=2;
        f.maximum=hi/d.scene.ell;
        if (!isfinite(f.maximum) || f.maximum<0 || lo>1e-10) { f.error=1; f.status=3; }
        if (f.status==0) {
            w.cell=cellAt(f,0,true,d);
            f.birthMean=gradientAt(f.origin,f.origin+f.direction*(.5*(w.cell.lo+w.cell.hi)*d.scene.ell),d,f.error);
            const double b=w.cell.value(0),derivative=w.cell.derivative(0);
            const double outward=d.scene.ell/d.scene.sigma*dot(f.direction,f.birthGradient);
            if (f.surface && !(outward>0)) f.error=4;
            const double entry[4]={f.surface?1.:0.,b,f.surface?-b:0.,f.surface?outward-derivative:0.};
            for (int j=0;j<4;++j) w.x[j]=float(asinh(entry[j]));
            f.remaining=f.queryOnly ? CUDART_INF : -log(f.uniforms[0]);
            if (f.error) f.status=3;
        }
    }
    syncGroup();
    if (w.f.status) return;
    dense(d.net.initial0,w.x,w.y,d,lane,true); dense(d.net.initial2,w.y,w.h,d,lane);
}
template<int H> __device__ void mixtureAndGradient(Work<H>& w,const Device& d,int lane) {
    const int h=d.net.hidden,k=d.net.components;
    if (lane==0) {
        // A sub-ulp fragment can round the inverse to its left endpoint. Match
        // segmentAt's left-limit derivative there, including the normal cell.
        if (w.f.q>0 && w.f.q==w.cell.lo) w.cell=cellAt(w.f,w.f.q,false,d);
        for (int i=0;i<h;++i) w.x[i]=w.h[i];
        w.x[h]=float(clamp((w.f.q-w.left)/d.net.step,0,1)); w.x[h+1]=float(log1p(w.f.q));
        w.x[h+2]=float(asinh(w.cell.value(w.f.q))); w.x[h+3]=float(asinh(w.cell.derivative(w.f.q)));
    }
    syncGroup();
    dense(d.net.mixture0,w.x,w.y,d,lane,true); dense(d.net.mixture2,w.y,w.x,d,lane,true);
    dense(d.net.mixture4,w.x,w.y,d,lane);
    if (lane==0) {
        double maximum=-CUDART_INF,sum=0,weights[16];
        for (int i=0;i<k;++i) maximum=fmax(maximum,double(w.y[i]));
        for (int i=0;i<k;++i) { weights[i]=exp(double(w.y[i])-maximum); sum+=weights[i]; }
        int j=0; double c=weights[0]/sum;
        while (j+1<k && w.f.uniforms[1]>=c) c+=weights[++j]/sum;
        const float raw=w.y[2*k+j];
        const double scale=double((raw>20.f ? raw : log1pf(expf(raw)))+.01f);
        const double mean=-w.cell.derivative(w.f.q)+double(w.y[k+j]);
        w.f.speed=fmax(0.,mean-scale*quantileLog(log1p(-w.f.uniforms[2])+logCdf(mean/scale)));
        const V3 p=clip(w.f.origin+w.f.direction*(w.f.q*d.scene.ell),d.scene);
        const V3 g=gradientAt(p,w.f.origin+w.f.direction*(.5*(w.cell.lo+w.cell.hi)*d.scene.ell),d,w.f.error);
        V3 u,v; frame(w.f.direction,u,v);
        const double rho=w.f.surface ? exp(-.5*w.f.q*w.f.q) : 0;
        const double variance=w.f.surface ? -expm1(-w.f.q*w.f.q) : 1;
        const V3 old=w.f.birthGradient-w.f.birthMean;
        const double sd=d.scene.sigma/d.scene.ell;
        w.f.gradient=w.f.direction*(-sd*w.f.speed)
            +u*(dot(u,g)+rho*dot(u,old)+sqrt(variance)*sd*w.f.normals[0])
            +v*(dot(v,g)+rho*dot(v,old)+sqrt(variance)*sd*w.f.normals[1]);
        if (!(w.f.speed>0) || !isfinite(dot(w.f.gradient,w.f.gradient)) || !(dot(w.f.direction,w.f.gradient)<0)) w.f.error=3;
        w.f.status=w.f.error ? 3 : 1;
    }
    syncGroup();
}
template<int H,bool Delta> __device__ void advance(Work<H>& w,const Device& d,int lane,
    curandStatePhilox4_32_10_t& rng) {
    if (lane==0) {
        w.left=w.f.cellIndex*d.net.step; w.right=fmin((w.f.cellIndex+1)*d.net.step,w.f.maximum);
        const Cell c=cellAt(w.f,w.left,w.left==0,d);
        for (int i=0;i<d.net.hidden;++i) w.x[i]=w.h[i];
        w.x[d.net.hidden]=float(asinh(c.value(w.left)));
        w.x[d.net.hidden+1]=float(asinh(c.derivative(w.left)));
        w.x[d.net.hidden+2]=float(log1p(w.left));
        ++w.f.intervals;
    }
    syncGroup();
    dense(d.net.head0,w.x,w.y,d,lane,true); dense(d.net.head2,w.y,w.x,d,lane);
    if (lane==0) { w.factor=exp(double(w.x[0])); w.cursor=w.left; }
    syncGroup();
    if constexpr (Delta) {
        // No synthetic observation or recurrent update at a null collision.
        // Carry one exponential optical depth across all majorant boundaries.
        while (w.cursor<w.right) {
            if (lane==0) {
                if (w.cursor==w.left || w.cursor>=w.cell.hi || w.cursor<w.cell.lo)
                    w.cell=cellAt(w.f,w.cursor,true,d);
                w.limit=fmin(w.right,w.cell.hi);
                const auto lower=causal_majorant::cubicLowerBounds(
                    w.cell.a,w.cell.b,w.cell.c,w.cell.d,w.cell.hi-w.cell.lo);
                const double upper=causal_majorant::localRateUpper(lower.value,lower.derivative);
                const double majorant=nextafter(w.factor*upper,CUDART_INF);
                ++w.f.bounds;
                if (!(w.limit>w.cursor) || !(w.factor>0) || !isfinite(w.factor) ||
                    !(majorant>0) || !isfinite(majorant)) { w.f.error=5; w.f.status=3; }
                while (!w.f.status && w.cursor<w.limit) {
                    const double mass=majorant*(w.limit-w.cursor);
                    if (w.f.remaining>=mass) { w.f.remaining-=mass; w.cursor=w.limit; break; }
                    const double q=w.cursor+w.f.remaining/majorant;
                    // Never silently discard a candidate rounded onto a boundary.
                    if (!(q>=w.cursor && q<w.limit)) { w.f.error=6; w.f.status=3; break; }
                    ++w.f.candidates;
                    const double actual=w.factor*rate(w.cell.value(q),w.cell.derivative(q));
                    if (!(actual>=0 && actual<=majorant)) { w.f.error=7; w.f.status=3; break; }
                    if (uniform(rng)<actual/majorant) { w.f.q=q; w.f.status=1; break; }
                    ++w.f.nulls; w.cursor=q; w.f.remaining=-log(uniform(rng));
                }
            }
            syncGroup();
            if (w.f.status==3) return;
            if (w.f.status==1) { mixtureAndGradient(w,d,lane); return; }
        }
    } else {
    while (w.cursor<w.right) {
        if (lane==0) {
            if (w.cursor==w.left || w.cursor>=w.cell.hi || w.cursor<w.cell.lo)
                w.cell=cellAt(w.f,w.cursor,true,d);
            double fine=(floor(w.cursor/d.net.fine)+1)*d.net.fine;
            if (!(fine>w.cursor)) fine+=d.net.fine;
            w.limit=fmin(w.right,fmin(w.cell.hi,fine));
            if (!(w.limit>w.cursor) || !isfinite(w.factor) || !(w.factor>0)) { w.f.error=3; w.f.status=3; }
        }
        syncGroup();
        if (w.f.status) return;
        integrate(w,w.cursor,w.limit,lane);
        if (lane==0) {
            if (!isfinite(w.mass) || w.mass<0) { w.f.error=3; w.f.status=3; }
            else if (w.mass>0 && w.f.remaining<=w.factor*w.mass) {
                w.residual=clamp(w.f.remaining/w.factor,0,w.mass); w.lo=w.cursor; w.hi=w.limit; w.f.status=1;
            } else {
                w.f.remaining-=w.factor*w.mass; w.f.hazard+=w.factor*w.mass;
            }
        }
        syncGroup();
        if (w.f.status==3) return;
        if (w.f.status==1) {
            for (int i=0;i<80 && w.hi-w.lo>d.scene.tolerance/d.scene.ell;++i) {
                const double mid=(w.lo+w.hi)/2;
                integrate(w,w.cursor,mid,lane);
                if (lane==0) { if (w.mass<w.residual) w.lo=mid; else w.hi=mid; }
                syncGroup();
            }
            if (lane==0) w.f.q=(w.lo+w.hi)/2;
            syncGroup(); mixtureAndGradient(w,d,lane); return;
        }
        if (lane==0) w.cursor=w.limit;
        syncGroup();
    }
    }
    if (w.right==w.f.maximum) {
        if (lane==0) { w.f.status=2; w.f.q=w.f.maximum; }
        syncGroup(); return;
    }
    if (lane==0) {
        const double points[3]={w.left,(w.left+w.right)/2,w.right};
        for (int i=0;i<3;++i) {
            const Cell c=cellAt(w.f,points[i],points[i]==0,d);
            w.x[i]=float(asinh(c.value(points[i]))); w.x[i+3]=float(asinh(c.derivative(points[i])));
        }
        w.x[6]=float(log1p(w.right-w.left));
    }
    syncGroup();
    dense(d.net.segment0,w.x,w.y,d,lane,true); dense(d.net.segment2,w.y,w.x,d,lane,true);
    dense(d.net.gruInput,w.x,w.a,d,lane); dense(d.net.gruHidden,w.h,w.b,d,lane);
    for (int j=lane;j<d.net.hidden;j+=lanes) {
        const int h=d.net.hidden;
        const float r=1.f/(1.f+expf(-w.a[j]-w.b[j]));
        const float z=1.f/(1.f+expf(-w.a[j+h]-w.b[j+h]));
        const float n=tanhf(w.a[j+2*h]+r*w.b[j+2*h]);
        w.h[j]=(1-z)*n+z*w.h[j];
    }
    syncGroup();
    if (lane==0) {
        ++w.f.cellIndex;
        for (int i=0;i<d.net.hidden;++i) if (!isfinite(w.h[i])) { w.f.error=3; w.f.status=3; }
    }
    syncGroup();
}
__device__ double uniform(curandStatePhilox4_32_10_t& rng) {
    // curand_uniform_double is (0,1]; reject the closed endpoint.
    double u; do { u=curand_uniform_double(&rng); } while (!(u>0 && u<1)); return u;
}
__device__ V3 environment(V3 direction,const Scene& c) {
    if (c.white) return {1,1,1};
    const double up=.5*(unit(direction).z+1); return {.15+.85*up,.2+.7*up,.35+.55*up};
}
__device__ double fresnel(double cosine,double eta,double k) {
    const double c=clamp(fabs(cosine),0,1);
    if (c==0) return 1;
    // Real arithmetic form of the same complex-amplitude conductor Fresnel.
    const double nr=eta*eta-k*k,ni=2*eta*k;
    const double gr=nr-(1-c*c),ga=hypot(gr,ni);
    const double u=sqrt(fmax(0.,.5*(ga+gr))),v=copysign(sqrt(fmax(0.,.5*(ga-gr))),ni);
    const double rs=((c-u)*(c-u)+v*v)/((c+u)*(c+u)+v*v);
    const double rp=((nr*c-u)*(nr*c-u)+(ni*c-v)*(ni*c-v))/((nr*c+u)*(nr*c+u)+(ni*c+v)*(ni*c+v));
    return .5*(rs+rp);
}
__device__ void finish(Slot& slot,V3 radiance,V3* pixels,Statistics* stats,const Scene& c) {
    const auto pixel=slot.path/c.spp;
    atomicAdd(&pixels[pixel].x,radiance.x/c.spp); atomicAdd(&pixels[pixel].y,radiance.y/c.spp);
    atomicAdd(&pixels[pixel].z,radiance.z/c.spp);
    atomicAdd(&stats->depth,static_cast<unsigned long long>(slot.depth));
    atomicAdd(&stats->flights,slot.flights); atomicAdd(&stats->intervals,slot.totalIntervals);
    atomicAdd(&stats->completed,1ULL); slot.active=0;
}
template<int H,bool Delta> __global__ __launch_bounds__(threads,4) void renderStep(Device d,Slot* slots,int capacity,V3* pixels,Statistics* stats,unsigned long long* next) {
    const int id=(blockIdx.x*blockDim.x+threadIdx.x)/lanes,lane=threadIdx.x%lanes;
    if (id>=capacity) return;
    __shared__ Work<H> work[groups]; Work<H>& w=work[threadIdx.x/lanes]; Slot& slot=slots[id];
    if (lane==0) w.f=slot.flight;
    for (int j=lane;j<d.net.hidden;j+=lanes) w.h[j]=slot.hidden[j];
    syncGroup();
    // Bounded launches keep the Windows display GPU responsive. Finished slots
    // immediately fetch a new independent camera path without host queues.
    for (int iteration=0;iteration<8;++iteration) {
        if (!slot.active) {
            if (lane==0) {
                slot.path=atomicAdd(next,1ULL);
                if (slot.path<static_cast<unsigned long long>(d.scene.width)*d.scene.height*d.scene.spp) {
                    slot.active=1; slot.depth=0; slot.totalIntervals=0; slot.flights=0; slot.throughput={1,1,1};
                    curand_init(d.scene.seed,slot.path,0,&slot.rng);
                    const auto pixel=slot.path/d.scene.spp;
                    const double x=(2*((pixel%d.scene.width+uniform(slot.rng))/d.scene.width)-1)*d.scene.aspect*d.scene.cameraScale;
                    const double y=(1-2*((pixel/d.scene.width+uniform(slot.rng))/d.scene.height))*d.scene.cameraScale;
                    w.f={}; w.f.origin=d.scene.camera; w.f.direction=unit(d.scene.forward+d.scene.right*x+d.scene.up*y);
                    double lo,hi;
                    if (!intersect(w.f.origin,w.f.direction,d.scene,lo,hi)) {
                        atomicAdd(&stats->escaped,1ULL); finish(slot,environment(w.f.direction,d.scene),pixels,stats,d.scene);
                    } else { w.f.origin=clip(w.f.origin+w.f.direction*lo,d.scene); slot.needsBirth=1; }
                }
            }
            syncGroup();
            if (!slot.active) continue;
        }
        if (slot.needsBirth) {
            if (lane==0) {
                for (int i=0;i<3;++i) w.f.uniforms[i]=uniform(slot.rng);
                // Draws are independent of the event even for an escaping ray.
                const double2 z=curand_normal2_double(&slot.rng); w.f.normals[0]=z.x; w.f.normals[1]=z.y;
                slot.needsBirth=0; ++slot.flights;
            }
            syncGroup(); initialize(w,d,lane);
        }
        if (!w.f.status) advance<H,Delta>(w,d,lane,slot.rng);
        if (lane==0 && w.f.status) {
            slot.totalIntervals+=w.f.intervals;
            if constexpr (Delta) {
                atomicAdd(&stats->candidates,w.f.candidates); atomicAdd(&stats->nulls,w.f.nulls);
                atomicAdd(&stats->bounds,w.f.bounds);
            }
            if (w.f.status==3) { atomicAdd(&stats->errors,1ULL); finish(slot,{0,0,0},pixels,stats,d.scene); }
            else if (w.f.status==2) {
                atomicAdd(&stats->escaped,1ULL); finish(slot,product(slot.throughput,environment(w.f.direction,d.scene)),pixels,stats,d.scene);
            } else {
                ++slot.depth; atomicAdd(&stats->collisions,1ULL);
                const V3 normal=unit(w.f.gradient);
                const double cosine=-dot(w.f.direction,normal);
                if (!d.scene.unitFresnel) slot.throughput=product(slot.throughput,{
                    fresnel(cosine,d.scene.eta.x,d.scene.k.x),fresnel(cosine,d.scene.eta.y,d.scene.k.y),fresnel(cosine,d.scene.eta.z,d.scene.k.z)});
                w.f.origin=clip(w.f.origin+w.f.direction*(w.f.q*d.scene.ell),d.scene);
                w.f.direction=unit(w.f.direction-normal*(2*dot(w.f.direction,normal)));
                w.f.birthGradient=w.f.gradient; w.f.surface=1; slot.needsBirth=1;
                if (slot.depth>=d.scene.rouletteDepth) {
                    const double p=clamp(fmax(slot.throughput.x,fmax(slot.throughput.y,slot.throughput.z)),.05,.95);
                    if (uniform(slot.rng)>=p) { atomicAdd(&stats->roulette,1ULL); finish(slot,{0,0,0},pixels,stats,d.scene); }
                    else slot.throughput=slot.throughput*(1/p);
                }
                if (slot.active && slot.depth>=d.scene.safetyDepth) {
                    atomicAdd(&stats->capped,1ULL); finish(slot,{0,0,0},pixels,stats,d.scene);
                }
            }
        }
        syncGroup();
    }
    if (lane==0) slot.flight=w.f;
    for (int j=lane;j<d.net.hidden;j+=lanes) slot.hidden[j]=w.h[j];
}
__global__ void sampleInit(const Input* inputs,Slot* slots,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    Slot& s=slots[i]; const Input& r=inputs[i]; s={}; s.active=1; s.needsBirth=1;
    s.flight.origin=r.origin; s.flight.direction=r.direction; s.flight.birthGradient=r.birthGradient;
    s.flight.surface=r.surface; s.flight.queryOnly=r.queryOnly;
    curand_init(r.trackingSeed,0,0,&s.rng);
    for (int j=0;j<3;++j) s.flight.uniforms[j]=r.uniforms[j];
    for (int j=0;j<2;++j) s.flight.normals[j]=r.normals[j];
}
template<int H,bool Delta> __global__ __launch_bounds__(threads,4) void sampleStep(Device d,Slot* slots,Output* output,int count,unsigned long long* completed) {
    const int id=(blockIdx.x*blockDim.x+threadIdx.x)/lanes,lane=threadIdx.x%lanes;
    if (id>=count || !slots[id].active) return;
    if (d.scene.deltaTracking && (slots[id].flight.queryOnly==Delta)) return;
    __shared__ Work<H> work[groups]; Work<H>& w=work[threadIdx.x/lanes]; Slot& s=slots[id];
    if (lane==0) w.f=s.flight;
    for (int j=lane;j<d.net.hidden;j+=lanes) w.h[j]=s.hidden[j];
    syncGroup();
    if (s.needsBirth) { initialize(w,d,lane); if (lane==0) s.needsBirth=0; syncGroup(); }
    for (int i=0;i<8 && !w.f.status;++i) advance<H,Delta>(w,d,lane,s.rng);
    if (lane==0) {
        s.flight=w.f;
        if (w.f.status) {
            output[id]={w.f.gradient,w.f.q,w.f.speed,w.f.hazard,w.f.intervals,w.f.status==1,w.f.error,
                w.f.candidates,w.f.nulls,w.f.bounds};
            s.active=0; atomicAdd(completed,1ULL);
        }
    }
    for (int j=lane;j<d.net.hidden;j+=lanes) s.hidden[j]=w.h[j];
}
void check(cudaError_t error,const char* operation) {
    if (error!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
template<class T> struct Buffer {
    T* p=nullptr;
    explicit Buffer(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(T)),"cudaMalloc"); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&)=delete;
    void zero() = delete;
};
} // namespace
struct Context::Impl {
    Buffer<unsigned char> grid;
    Buffer<float> weights;
    Device device{};
    cudaStream_t stream=nullptr;
    Impl(const void* data,std::size_t size,const Network& n,const std::vector<float>& w,const Scene& s)
        :grid(size),weights(w.size()) {
        check(cudaMemcpy(grid.p,data,size,cudaMemcpyHostToDevice),"upload NanoVDB");
        check(cudaMemcpy(weights.p,w.data(),w.size()*sizeof(float),cudaMemcpyHostToDevice),"upload joint weights");
        const auto* host=static_cast<const nanovdb::NanoGrid<float>*>(data);
        const auto o=host->map().applyMap(nanovdb::Vec3d(0.));
        device={reinterpret_cast<const nanovdb::NanoGrid<float>*>(grid.p),weights.p,n,s,{o[0],o[1],o[2]},host->voxelSize()[0]};
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create CUDA stream");
    }
    ~Impl() { if (stream) cudaStreamDestroy(stream); }
};
Context::Context(const void* grid,std::size_t bytes,const Network& n,const std::vector<float>& w,const Scene& s)
    :impl_(new Impl(grid,bytes,n,w,s)) {}
Context::~Context() { delete impl_; }
std::vector<Output> Context::sample(const std::vector<Input>& inputs) {
    std::vector<Output> output(inputs.size()); if (inputs.empty()) return output;
    auto& c=*impl_;
    // Bound temporary storage independently of a diagnostic's total sample count.
    constexpr int batch=8192;
    Buffer<Input> in(batch); Buffer<Slot> slots(batch); Buffer<Output> out(batch); Buffer<unsigned long long> done(1);
    for (std::size_t offset=0;offset<inputs.size();offset+=batch) {
        const int count=int(std::min<std::size_t>(batch,inputs.size()-offset));
        check(cudaMemcpyAsync(in.p,inputs.data()+offset,count*sizeof(Input),cudaMemcpyHostToDevice,c.stream),"upload flight inputs");
        check(cudaMemsetAsync(done.p,0,sizeof(unsigned long long),c.stream),"clear query counter");
        sampleInit<<<(count+127)/128,128,0,c.stream>>>(in.p,slots.p,count);
        unsigned long long completed=0;
        for (unsigned int launches=0;completed<static_cast<unsigned>(count);++launches) {
            if (launches>1000000) throw std::runtime_error("CUDA flight iteration limit");
            if (c.device.net.hidden<=16) {
                sampleStep<16,false><<<(count+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,out.p,count,done.p);
                if (c.device.scene.deltaTracking)
                    sampleStep<16,true><<<(count+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,out.p,count,done.p);
            } else {
                sampleStep<32,false><<<(count+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,out.p,count,done.p);
                if (c.device.scene.deltaTracking)
                    sampleStep<32,true><<<(count+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,out.p,count,done.p);
            }
            check(cudaGetLastError(),"launch flight sampler");
            if (launches%8==7) {
                check(cudaMemcpyAsync(&completed,done.p,sizeof(completed),cudaMemcpyDeviceToHost,c.stream),"read query counter");
                check(cudaStreamSynchronize(c.stream),"flight sampling");
            }
        }
        check(cudaMemcpyAsync(output.data()+offset,out.p,count*sizeof(Output),cudaMemcpyDeviceToHost,c.stream),"read flight samples");
        check(cudaStreamSynchronize(c.stream),"read flight results");
    }
    return output;
}
std::vector<V3> Context::render(int capacity,Statistics& statistics,const std::function<void(std::uint64_t)>& progress) {
    auto& c=*impl_; const auto pixels=std::size_t(c.device.scene.width)*c.device.scene.height;
    const auto paths=static_cast<unsigned long long>(pixels)*c.device.scene.spp;
    capacity=int(std::min<unsigned long long>(capacity,paths));
    Buffer<Slot> slots(capacity); Buffer<V3> image(pixels); Buffer<Statistics> stats(1); Buffer<unsigned long long> next(1);
    check(cudaMemsetAsync(slots.p,0,capacity*sizeof(Slot),c.stream),"clear ray pool");
    check(cudaMemsetAsync(image.p,0,pixels*sizeof(V3),c.stream),"clear image");
    check(cudaMemsetAsync(stats.p,0,sizeof(Statistics),c.stream),"clear counters");
    check(cudaMemsetAsync(next.p,0,sizeof(unsigned long long),c.stream),"clear path queue");
    statistics={}; unsigned long long kernels=0;
    while (statistics.completed<paths) {
        for (int i=0;i<8;++i) {
            if (c.device.net.hidden<=16) {
                if (c.device.scene.deltaTracking)
                    renderStep<16,true><<<(capacity+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,capacity,image.p,stats.p,next.p);
                else renderStep<16,false><<<(capacity+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,capacity,image.p,stats.p,next.p);
            } else {
                if (c.device.scene.deltaTracking)
                    renderStep<32,true><<<(capacity+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,capacity,image.p,stats.p,next.p);
                else renderStep<32,false><<<(capacity+groups-1)/groups,threads,0,c.stream>>>(c.device,slots.p,capacity,image.p,stats.p,next.p);
            }
            check(cudaGetLastError(),"launch joint renderer"); ++kernels;
        }
        check(cudaMemcpyAsync(&statistics,stats.p,sizeof(Statistics),cudaMemcpyDeviceToHost,c.stream),"read render counters");
        check(cudaStreamSynchronize(c.stream),"joint GPU render");
        if (statistics.errors) throw std::runtime_error("CUDA renderer encountered "+std::to_string(statistics.errors)+" invalid flights; refusing a biased image");
        progress(statistics.completed);
    }
    statistics.kernels=kernels;
    std::vector<V3> result(pixels);
    check(cudaMemcpyAsync(result.data(),image.p,pixels*sizeof(V3),cudaMemcpyDeviceToHost,c.stream),"read image");
    check(cudaStreamSynchronize(c.stream),"finish joint GPU render");
    return result;
}
} // namespace mf::cuda_renewal
