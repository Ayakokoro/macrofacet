#include "TestHarness.h"
#include "macrofacet/gpss/Matern32Reference.h"
#include "macrofacet/gpss/CovarianceKernel.h"
#include "macrofacet/experiments/FirstPassageExperiment.h"
#include "macrofacet/experiments/FirstPassageRice.h"
#include <Eigen/Eigenvalues>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
class PointProbeMean final : public mf::MeanField {
public:
    mutable int queries = 0;
    mutable double furthest = 0;
    const char* typeName() const override { return "point_probe"; }
    mf::MeanJet evaluate(const mf::Point3& p) const override {
        ++queries; furthest = std::max(furthest,p.x());
        return {1+p.x()+2*p.x()*p.x(),mf::Vector3(1+4*p.x(),0,0)};
    }
    mf::BoundsSummary bounds(const mf::Bounds3&) const override { return {}; }
};
class PiecewiseMean final : public mf::MeanField {
public:
    const char* typeName() const override { return "piecewise_test"; }
    mf::MeanJet evaluate(const mf::Point3& p) const override {
        const double x=p.x();
        return {0.2+x-2*x*x+x*x*x+0.7*std::max(0.0,x-0.3),
                mf::Vector3(1-4*x+3*x*x+(x>=0.3 ? 0.7 : 0.0),0,0)};
    }
    mf::BoundsSummary bounds(const mf::Bounds3&) const override { return {}; }
    void appendRayBreakpoints(const mf::Point3& origin,const mf::Vector3& direction,
        double lo,double hi,std::vector<double>& knots) const override {
        const double t=(0.3-origin.x())/direction.x();
        if (t>lo && t<hi) knots.push_back(t);
    }
};
}

void testRenewalReference(TestContext& context) {
    using namespace mf;
    const auto probe = std::make_shared<PointProbeMean>();
    const auto lazy = RayMeanProfile::pointLinear(probe,Point3::Zero(),Vector3::UnitX(),0.5,0.4,0.8,0.25);
    context.require(probe->queries == 1 && lazy.constructedSegmentCount() == 1 && probe->furthest == 0,
                    "point profile queries only the current start, never future endpoints");
    context.near(lazy.maximumX(),2,0,"lazy profile knows its extent without materialization");
    const auto firstFeatures = lazy.segment(0).features();
    context.near(firstFeatures[0],2,0,"point features normalize current mean");
    context.near(firstFeatures[1],2.2,1e-14,"point endpoint is tangent extrapolation");
    context.near(firstFeatures[2],0.2,1e-14,"point features normalize directional slope");
    context.require(probe->queries == 1,"reusing a point segment does not query the field again");
    auto independent = lazy;
    context.require(independent.hasSegment(1) && lazy.constructedSegmentCount() == 1 && probe->queries == 2,
                    "copied point profiles have independent lazy caches");
    context.near(probe->furthest,0.1,1e-14,"advancing queries only the next start");
    const auto plane = std::make_shared<PlaneMean>(Vector3::UnitX(),0.23);
    for (double sign : {-1.0,1.0}) {
        const Vector3 direction = sign*Vector3::UnitX();
        const auto linear = RayMeanProfile::pointLinear(plane,Point3(0.2,0,0),direction,0.7,0.4,0.8,0.25);
        for (const auto& segment : linear.segments()) {
            const double x = segment.begin+0.63*(segment.end-segment.begin);
            context.near(segment.value(x),plane->evaluate(Point3(0.2,0,0)+0.4*x*direction).value/0.7,
                         1e-13,"point tangent is exact for an affine field in either direction");
            context.near(segment.derivative(x),0.4/0.7*sign,1e-13,"point slope has correct physical scaling");
            context.require(segment.polynomial.a == 0 && segment.polynomial.b == 0,
                            "point mode never recovers quadratic or cubic coefficients");
        }
    }
    const PiecewiseMean field;
    const double sigma=0.7,ell=0.4;
    const auto profile=RayMeanProfile::fromField(field,Point3::Zero(),Vector3::UnitX(),
                                               sigma,ell,0.8,0.25);
    bool foundBoundary=false;
    for (const auto& segment : profile.segments()) {
        for (double u : {0.0,0.17,0.63,1.0}) {
            const double x=segment.begin+u*(segment.end-segment.begin);
            context.near(segment.value(x),field.evaluate(Point3(ell*x,0,0)).value/sigma,
                         1e-13,"ray profile preserves the full cell cubic");
        }
        const auto f=segment.features();
        const auto recovered=hermitePolynomial(f[0],f[2]/std::exp(f[4]),f[1],
                                               f[3]/std::exp(f[4]),std::exp(f[4]));
        context.near(recovered.value(0.37),segment.polynomial.value(0.37),1e-13,
                     "five features recover the complete mean segment");
        if (std::abs(segment.end-0.3/ell)<1e-14) {
            foundBoundary=true;
            context.near(segment.derivative(segment.end),ell/sigma*(1-4*0.3+3*0.09),
                         1e-12,"left endpoint derivative remains inside the left cell");
        }
    }
    context.require(foundBoundary,"profile preserves interpolation boundaries");
    const auto refined=profile.subdivided(0.03);
    for (const auto& segment : refined.segments()) {
        const double x=0.5*(segment.begin+segment.end);
        context.near(segment.value(x),field.evaluate(Point3(ell*x,0,0)).value/sigma,
                     2e-13,"reference subdivision does not change deterministic geometry");
    }
    const auto reverse=RayMeanProfile::fromField(field,Point3(0.8,0,0),-Vector3::UnitX(),
                                                sigma,ell,0.8,0.25);
    for (const auto& segment : reverse.segments()) {
        const double x=0.5*(segment.begin+segment.end);
        const auto jet=field.evaluate(Point3(0.8-ell*x,0,0));
        context.near(segment.value(x),jet.value/sigma,1e-13,"reverse ray mean");
        context.near(segment.derivative(x),-ell*jet.gradient.x()/sigma,1e-12,
                     "reverse ray derivative units");
    }
    const auto crossing=firstHermiteDowncrossing({0,1,-0.6,0.08},1.0,1e-12);
    context.require(crossing.found,"same-sign interval endpoints can hide a first crossing");
    context.near(crossing.fraction,0.2,1e-11,"earliest of multiple polynomial roots is used");
    context.near(crossing.derivative,-0.2,1e-11,"crossing speed uses the same polynomial");
    const auto tinyCrossing=firstHermiteDowncrossing({0,1e-20,-0.6e-20,0.08e-20},1.0,1e-12);
    context.require(tinyCrossing.found,"small amplitude does not hide interior crossings");
    context.near(tinyCrossing.fraction,0.2,1e-11,"crossing roots are independent of field scale");
    context.require(!firstHermiteDowncrossing({0,1,-1,0.25},1.0,1e-12).found,
                    "touching zero at a stationary point is not a downcrossing");

    const auto kernel=CovarianceKernel::fromCorrelationLengths(
        CovarianceKernelType::Matern32,2.0,Vector3::Constant(0.4));
    const FirstPassageStationaryKernel scalar({"m32","matern_3_2",4.0,0.4,1.0});
    for (double x : {0.0,0.01,0.4,1.7}) {
        const auto jet=kernel.evaluate(Point3(x,0,0),Point3::Zero());
        const double q=x/0.4;
        context.near(jet.valueValue,4*(1+q)*std::exp(-q),1e-13,"unit-decay Matern covariance");
        context.near(jet.valueValue,scalar.covariance(x),1e-13,"3D and reference kernels agree");
        context.near(jet.gradientXValueY.x(),scalar.firstDerivative(x),1e-12,"kernel first derivative agrees");
        context.near(jet.gradientXGradientY(0,0),-scalar.secondDerivative(x),1e-12,"kernel second derivative agrees");
    }
    context.near(scalar.derivativeVariance(),25.0,1e-12,"Matern beta equals one");
    const auto t1=matern32Transition(0.13),t2=matern32Transition(0.27),t=matern32Transition(0.4);
    context.near((t.matrix-t2.matrix*t1.matrix).norm(),0,1e-14,"transition semigroup");
    context.near((t.covariance-t2.covariance-t2.matrix*t1.covariance*t2.matrix.transpose()).norm(),
                 0,1e-14,"transition covariance composition");
    for (double step : {1e-10,1e-7,1e-3,0.2,5.0}) {
        const auto transition=matern32Transition(step);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(transition.covariance);
        context.require(eig.eigenvalues().minCoeff()>=0,"small-step reference covariance is PSD");
        context.near((transition.covariance+transition.matrix*transition.matrix.transpose()
                      -Eigen::Matrix2d::Identity()).norm(),0,1e-14,"unit stationary state covariance");
    }
    const auto flat=RayMeanProfile::affine(0,0,2.0);
    Random rng(7231);
    double sumZ=0,sumD=0,sumD2=0;
    constexpr int count=12000;
    for (int i=0;i<count;++i) {
        const auto state=initializeMatern32(flat,{},rng);
        context.require(state[0]>0,"mode A starts strictly in the positive domain");
        sumZ+=state[0]; sumD+=state[1]; sumD2+=state[1]*state[1];
    }
    context.near(sumZ/count,std::sqrt(2.0/kPi),0.02,"mode A truncated initial value mean");
    context.near(sumD/count,0,0.035,"mode A initial derivative has zero mean");
    context.near(sumD2/count,1,0.05,"mode A initial derivative has unit variance");
    const auto affine=RayMeanProfile::affine(0.3,-0.2,2.0);
    const auto birth=initializeMatern32(affine,{RayStartMode::SurfaceOutward,0.7},rng);
    context.near(birth[0],-0.3,1e-15,"mode B pins the surface value");
    context.near(birth[1],0.9,1e-15,"mode B removes the actual mean derivative");
    for (RayStartMode mode : {RayStartMode::PositiveExterior,RayStartMode::SurfaceOutward}) {
        int hits[2]{0,0};
        for (int resolution=0;resolution<2;++resolution) {
            FirstPassageReferenceSettings settings;
            settings.step=resolution==0 ? 0.0625 : 0.03125;
            settings.minimumStep=settings.step/16;
            for (int trial=0;trial<1024;++trial) {
                Random draw(712367+trial+resolution*100000);
                const auto sample=sampleMatern32FirstPassage(affine,{mode,0.7},settings,draw);
                hits[resolution]+=sample.hit;
                context.require(sample.distance>0 && sample.distance<=2.0 &&
                    (sample.hit ? sample.speed>0 : std::isnan(sample.speed)),
                    "reference samples preserve censoring and downward-crossing support");
            }
        }
        context.near(hits[0]/1024.0,hits[1]/1024.0,0.07,"reference escape converges under step refinement");
    }

    // Exercise the public CLI schema, field path, A/B initialization and exports.
    namespace fs=std::filesystem;
    const auto directory=fs::temp_directory_path()/"macrofacet_renewal_reference_test";
    fs::create_directories(directory);
    nlohmann::json root={{"schema_version",1},{"seed",7231},
        {"output_directory",directory.string()},
        {"first_passage",{
            {"process",{{"mean_field",{{"mean_type","plane"},{"plane_normal",{0,0,1}},
                {"plane_offset",0},{"sigma",0.1}}}}},
            {"initial_condition",{{"type","positive_exterior"},{"rays",{
                {{"id","exterior"},{"origin",{0,0,0.1}},{"direction",{1,0,-0.2}}}}}}},
            {"grid",{{"max_time",1.0},{"step_sizes",{0.0625,0.03125}}}},
            {"profile",{{"maximum_step",0.25}}},
            {"monte_carlo",{{"trajectories",32},{"thread_count",1},{"write_raw_samples",true}}},
            {"curve",{{"bins",8}}},
            {"kernels",{{{"id","m32"},{"type","matern_3_2"},{"variance",0.01},{"length_scale",0.2}}}}
        }}};
    const auto path=directory/"input.json";
    { std::ofstream out(path); out<<root; }
    auto config=loadFirstPassageExperimentConfig(path);
    runFirstPassageExperiment(config);
    context.require(fs::exists(directory/"first_passage_mean_segments.csv"),"common mean features are exported");
    const auto read=[](const fs::path& file) { std::ifstream in(file); std::ostringstream out; out<<in.rdbuf(); return out.str(); };
    const auto serial=read(directory/"first_passage_samples.csv");
    config.threadCount=3;
    runFirstPassageExperiment(config);
    context.require(serial==read(directory/"first_passage_samples.csv"),"mode A reference is reproducible across threads");
    auto roundtrip=loadFirstPassageExperimentConfig(directory/"resolved_first_passage_config.json");
    context.require(roundtrip.initialConditionType=="positive_exterior" &&
        roundtrip.collisionStates.size()==1,"mode A known conditioning round-trips");
    auto seRoot = root;
    seRoot["first_passage"]["kernels"][0]["type"]="squared_exponential";
    { std::ofstream out(path); out<<seRoot; }
    auto seConfig=loadFirstPassageExperimentConfig(path);
    runFirstPassageExperiment(seConfig);
    const auto seSerial=read(directory/"first_passage_samples.csv");
    seConfig.threadCount=3;
    runFirstPassageExperiment(seConfig);
    context.require(seSerial==read(directory/"first_passage_samples.csv"),
                    "SE exterior reference is reproducible across threads");
    context.require(seSerial.find("conditioned_grid_circulant")!=std::string::npos &&
                    seSerial.find("grid_residual_hermite_cellwise_mean")!=std::string::npos,
                    "SE exterior reference uses conditioned grid with complete mean");
    // For a zero-mean SE ray, the endpoint crossing probability conditional on
    // F(0)>0 is acos(rho(t))/pi. Over a short interval extra recrossings are rare.
    // This catches accidentally treating exterior starts as F(0)=F'(0)=0.
    seConfig.processMeanField.reset();
    seConfig.collisionStates[0].ray.reset();
    seConfig.collisionStates[0].beta0=0;
    seConfig.collisionStates[0].betaMeanSlope=0;
    seConfig.maximumTime=0.25;
    seConfig.stepSizes={1.0/128};
    seConfig.trajectories=4096;
    runFirstPassageExperiment(seConfig);
    std::istringstream rows(read(directory/"first_passage_samples.csv"));
    std::string row,cell;
    std::getline(rows,row);
    std::istringstream header(row);
    int eventColumn=-1,column=0;
    while (std::getline(header,cell,',')) { if (cell=="event") eventColumn=column; ++column; }
    int hits=0,seCount=0;
    while (std::getline(rows,row)) {
        std::istringstream fields(row); column=0;
        while (std::getline(fields,cell,',')) { if (column==eventColumn) hits+=std::stoi(cell); ++column; }
        ++seCount;
    }
    context.require(seCount==4096 && eventColumn>=0,"SE preserves every trajectory including censored samples");
    context.near(hits/4096.0,std::acos(std::exp(-0.5*0.25*0.25))/kPi,0.016,
                 "SE exterior short-horizon hits agree with Gaussian endpoint probability");
    root["first_passage"]["initial_condition"]["type"]="collision_state";
    root["first_passage"]["initial_condition"]["rays"][0]["gradient"]={1,0,0};
    { std::ofstream out(path); out<<root; }
    runFirstPassageExperiment(loadFirstPassageExperimentConfig(path));
    context.require(read(directory/"first_passage_samples.csv").find("state_bridge_hermite_cellwise_mean")
        !=std::string::npos,"spatial Matern mean uses the state-space reference backend");
    fs::remove_all(directory);
}
