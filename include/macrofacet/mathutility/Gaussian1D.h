#pragma once

#include "macrofacet/core/Random.h"

namespace mf {

double normalPdf(double z);
double normalLogPdf(double z);
double normalCdf(double z);
double normalLogCdf(double z);
double normalSurvival(double z);
double normalLogSurvival(double z);
double normalPdfOverCdf(double z);
double normalQuantile(double u);
double normalQuantileFromLogCdf(double logU);
double sampleStandardNormal(Random& rng);

} // namespace mf

