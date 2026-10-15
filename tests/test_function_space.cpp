#include "TestHarness.h"
#include "macrofacet/gpss/FunctionSpaceRaySampler.h"
#include <array>

void testFunctionSpace(TestContext& context) {
    using namespace mf;
    // Independent closed-form SE value/derivative moments conditioned on Z(0), Z'(0).
    FunctionSpaceRaySampler sampler({1,4,0.25,1e-10});
    constexpr int trials = 20000;
    std::array<double,4> sums{}, squares{};
    double product = 0.0;
    Random rng(578431);
    for (int i = 0; i < trials; ++i) {
        FunctionSpaceHistory history{{0.7},-0.4};
        const auto batch = sampler.next(history,rng);
        const std::array<double,4> draw{batch.front().value,batch.front().derivative,
                                      batch.back().value,batch.back().derivative};
        for (int j = 0; j < 4; ++j) { sums[j] += draw[j]; squares[j] += draw[j]*draw[j]; }
        product += draw[0]*draw[2];
    }
    for (int i = 0; i < 2; ++i) {
        const double t = i == 0 ? 0.25 : 1.0, r = std::exp(-0.5*t*t);
        context.near(sums[2*i]/trials,r*(0.7-0.4*t),0.014,"GPIS conditional value mean");
        context.near(sums[2*i+1]/trials,r*(-0.7*t-0.4*(1-t*t)),0.018,"GPIS analytic derivative mean");
        context.near(squares[2*i]/trials-std::pow(sums[2*i]/trials,2),
            1-(1+t*t)*r*r,0.014,"GPIS conditional value variance");
        context.near(squares[2*i+1]/trials-std::pow(sums[2*i+1]/trials,2),
            1-(t*t+std::pow(1-t*t,2))*r*r,0.025,"GPIS conditional derivative variance");
    }
    context.near(product/trials-sums[0]*sums[2]/(double(trials)*trials),
        std::exp(-0.5*0.75*0.75)-1.25*std::exp(-0.5*(1+0.25*0.25)),0.008,
        "GPIS batch values retain their joint conditional covariance");

    FunctionSpaceRaySampler longer({5,3,1.0/64,1e-10});
    FunctionSpaceHistory a{{0.2},0.1}, b=a;
    Random ra(39), rb(39);
    for (int block=0; block<20; ++block) {
        const auto x=longer.next(a,ra), y=longer.next(b,rb);
        context.require(a.values.size()==std::min<std::size_t>(5,1+3*(block+1)),
                        "GPIS keeps at most H values during warmup and continuation");
        context.near(a.derivative,x.back().derivative,0,"GPIS keeps only latest derivative");
        context.near(x.back().value,y.back().value,0,"GPIS seed reproduction");
    }
    // Geometry subdivisions and censoring horizons must never move the residual grid.
    FunctionSpaceRaySampler reference({1,8,1.0/64,1e-10,"hermite_residual_cellwise_mean"});
    const RayMeanProfile full({{0.0,1.1,hermitePolynomial(0.0,-0.15,0.4,0.6,1.1)},
                              {1.1,3.017,hermitePolynomial(0.4,-0.2,-0.5,0.1,1.917)}});
    const auto split=full.subdivided(0.073);
    const RayMeanProfile shortRay({full.segments().front().restricted(0.0,1.013)});
    int negative=0, hits=0;
    for (int i=0; i<256; ++i) {
        Random r1(i+498),r2(i+498),r3(i+498);
        const auto x=sampleFunctionSpaceFirstPassage(reference,full,FunctionSpaceStart::Unconditioned,0,r1);
        const auto y=sampleFunctionSpaceFirstPassage(reference,split,FunctionSpaceStart::Unconditioned,0,r2);
        const auto z=sampleFunctionSpaceFirstPassage(reference,shortRay,FunctionSpaceStart::Unconditioned,0,r3);
        negative += x.initialSign<0; hits += x.hit;
        context.require(x.hit==y.hit,"mean subdivision preserves first crossing");
        context.near(x.distance,y.distance,1e-9,"mean subdivision preserves hit distance");
        context.near(x.speed,y.speed,1e-8,"mean subdivision preserves crossing derivative");
        if (x.hit && x.distance<=shortRay.maximumX()) {
            context.require(z.hit,"short ray keeps hits before censoring");
            context.near(z.distance,x.distance,1e-9,"nonintegral horizon preserves sampled prefix");
        } else {
            context.require(!z.hit,"short ray censors later hits");
            context.near(z.distance,shortRay.maximumX(),0,"exact censoring horizon");
        }
        context.require(x.distance>0 && x.distance<=full.maximumX() && (!x.hit || x.speed>0),
                        "GPIS first-zero has no entry atom and positive crossing speed");
    }
    context.require(negative>95 && negative<160,"unconditioned entry retains both signs");
    context.require(hits>60 && hits<250,"GPIS first-zero produces hits and censored samples");
    for (const auto start : {FunctionSpaceStart::SurfaceOutward,FunctionSpaceStart::PositiveExterior}) {
        for (int i=0; i<64; ++i) {
            Random random(85+i);
            const auto x=sampleFunctionSpaceFirstPassage(reference,full,start,1.0,random);
            context.require(x.initialSign==1 && x.distance>0 && (!x.hit || x.speed>0),
                            "GPIS A/B start convention excludes entry collision");
        }
    }
    // Cancel one replayed residual at node values to prescribe total-field
    // secants. The first sign change straddles a batch boundary, and the mean
    // is curved and split independently of the sampling grid.
    FunctionSpaceRaySampler linear({1,2,1.0,1e-10});
    Random replay(10893);
    std::vector<FunctionSpaceState> nodes{{replay.standardNormal(),replay.standardNormal()}};
    FunctionSpaceHistory replayHistory{{nodes[0].value},nodes[0].derivative};
    for (int block=0; block<2; ++block) {
        const auto batch=linear.next(replayHistory,replay);
        nodes.insert(nodes.end(),batch.begin(),batch.end());
    }
    for (const double sign : {-1.0,1.0}) {
        const std::array<double,5> values{2,1,.5,-1.5,-2};
        std::vector<RayMeanSegment> segments;
        for (int j=0; j<4; ++j)
            segments.push_back({double(j),double(j+1),hermitePolynomial(
                sign*values[j]-nodes[j].value,3.0,
                sign*values[j+1]-nodes[j+1].value,-2.0,1.0)});
        const RayMeanProfile profile(std::move(segments));
        for (const auto& partition : {profile,profile.subdivided(.073)}) {
            Random random(10893);
            const auto x=sampleFunctionSpaceFirstPassage(linear,partition,FunctionSpaceStart::Unconditioned,0,random);
            context.require(x.hit && x.initialSign==int(sign),"linear detector preserves both entry signs and batch seams");
            context.near(x.distance,2.25,1e-12,"Tungsten total-field linear hit distance");
            context.near(x.speed,2.0,1e-12,"Tungsten total-field secant speed, not analytic derivative");
        }
    }
    // Equal-sign endpoints can conceal two Hermite roots. Linear detection
    // intentionally ignores them even if the geometry adds interior knots.
    const RayMeanProfile excursion({{0,1,hermitePolynomial(
        1-nodes[0].value,-8-nodes[0].derivative,
        1-nodes[1].value,8-nodes[1].derivative,1)}});
    FunctionSpaceRaySampler hermite({1,2,1.0,1e-10,"hermite_residual_cellwise_mean"});
    Random linearRandom(10893),hermiteRandom(10893);
    const auto missed=sampleFunctionSpaceFirstPassage(linear,excursion.subdivided(.05),
        FunctionSpaceStart::Unconditioned,0,linearRandom);
    const auto found=sampleFunctionSpaceFirstPassage(hermite,excursion,
        FunctionSpaceStart::Unconditioned,0,hermiteRandom);
    context.require(!missed.hit && found.hit,"linear and Hermite retain distinct crossing semantics");
    context.near(found.distance,(1-std::sqrt(.5))/2,1e-10,"Hermite interior root retained for comparison");

    // Clipped horizons have no out-of-domain mean. Reconstruct only the last
    // residual endpoint, then apply the same total-field secant rule.
    const auto residual=hermitePolynomial(nodes[0].value,nodes[0].derivative,
        nodes[1].value,nodes[1].derivative,1);
    for (const double target : {-2.0,2.0}) {
        const RayMeanProfile clipped({{0,.75,hermitePolynomial(
            1-nodes[0].value,3,target-residual.value(.75),-2,.75)}});
        Random random(10893);
        const auto x=sampleFunctionSpaceFirstPassage(linear,clipped,FunctionSpaceStart::Unconditioned,0,random);
        context.require(x.hit==(target<0),"linear clipped endpoint hit or censoring");
        context.near(x.distance,target<0 ? .25 : .75,1e-12,"linear clipped horizon distance");
        if (x.hit) context.near(x.speed,4.0,1e-12,"linear clipped endpoint secant");
    }
    const RayMeanProfile birth({{0,1,hermitePolynomial(0,0,-100,0,1)}});
    Random birthLinear(81),birthHermite(81);
    const auto discreteBirth=sampleFunctionSpaceFirstPassage(linear,birth,FunctionSpaceStart::SurfaceOutward,1,birthLinear);
    const auto smoothBirth=sampleFunctionSpaceFirstPassage(hermite,birth,FunctionSpaceStart::SurfaceOutward,1,birthHermite);
    context.require(!discreteBirth.hit && discreteBirth.distance==1 && smoothBirth.hit && smoothBirth.distance>0,
                    "linear ignores the birth zero; early sub-step returns require Hermite");

    for (const auto config : {FunctionSpaceSamplerConfig{0,8,.1,1e-10},
                             FunctionSpaceSamplerConfig{1,0,.1,1e-10},
                             FunctionSpaceSamplerConfig{1,8,0,1e-10},
                             FunctionSpaceSamplerConfig{1,8,.1,0},
                             FunctionSpaceSamplerConfig{1,8,.1,1e-10,"unknown"}}) {
        bool rejected=false;
        try { FunctionSpaceRaySampler invalid(config); }
        catch (const std::invalid_argument&) { rejected=true; }
        context.require(rejected,"invalid GPIS configuration rejected");
    }
}
