#pragma once

#include "macrofacet/core/Random.h"

namespace mf {

double normalPdf(double z);
double normalLogPdf(double z);
double normalCdf(double z);
double normalLogCdf(double z);
// p (Z > z)
double normalSurvival(double z);
double normalLogSurvival(double z);
double normalPdfOverCdf(double z);
// phi^{-1}(u) for 0<u<1
double normalQuantile(double u);
double normalQuantileFromLogCdf(double logU);
double sampleStandardNormal(Random& rng);

} // namespace mf

