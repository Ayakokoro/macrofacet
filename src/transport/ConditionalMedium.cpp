#include "macrofacet/transport/ConditionalMedium.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {

ConditionalMajorantCursor::ConditionalMajorantCursor(const ConditionalFlightKernel& flight,
    double maximumAge, DdaTrackingDiagnostics* diagnostics)
    : flight_(flight), diagnostics_(diagnostics),
      age_(flight.currentAge()),
      end_(maximumAge < 0.0 ? flight.maximumAgeInDomain() :
           std::min(maximumAge, flight.maximumAgeInDomain())),
      correlationLength_(1.0/std::sqrt(flight.state().direction.dot(
          flight.field().kernel.precision()*flight.state().direction))) {
    if (!(end_ >= age_))
        throw NumericError(NumericStatus::InvalidInput, "invalid conditional flight limit");
    if (end_ > age_) birth_ = flight_.birthMajorant(end_);
}

double ConditionalMajorantCursor::bound(double begin, double end) {
    if (diagnostics_) ++diagnostics_->boundIntervals;
    const double maximum = end <= birth_.smoothEnd
        ? flight_.birthAdjacentMajorant(begin,end,birth_)
        : flight_.intervalMajorant(begin,end);
    if (diagnostics_)
        diagnostics_->maximumConditionalIntervalMajorant = std::max(
            diagnostics_->maximumConditionalIntervalMajorant, maximum);
    return maximum;
}

std::optional<ExtinctionSegment> ConditionalMajorantCursor::next() {
    if (nearPending_) {
        nearPending_ = false;
        if (age_ < birth_.split) {
            const ExtinctionSegment segment{age_,birth_.split,birth_.maximum,true};
            age_=birth_.split;
            return segment;
        }
    }
    while (age_ < end_ || !pending_.empty()) {
        if (pending_.empty()) {
            const double lo=age_;
            double hi=std::min(end_,lo+std::min(0.5*lo,0.5*correlationLength_));
            if (lo < birth_.smoothEnd) hi=std::min(hi,birth_.smoothEnd);
            std::vector<double> knots;
            flight_.field().mean->appendRayBreakpoints(
                flight_.state().birthPosition,flight_.state().direction,lo,hi,knots);
            for (double knot : knots)
                if (knot>lo && knot<hi) hi=knot;
            if (!(hi>lo))
                throw NumericError(NumericStatus::InvalidMajorant,
                                   "conditional segment cannot advance");
            pending_.push_back({lo,hi,bound(lo,hi),false});
        }
        Pending interval=pending_.back();
        pending_.pop_back();
        if (interval.tighten)
            interval.maximum=std::min(interval.maximum,bound(interval.begin,interval.end));
        while (interval.maximum*(interval.end-interval.begin)>64.0) {
            const double middle=0.5*interval.begin+0.5*interval.end;
            if (!(middle>interval.begin && middle<interval.end)) break;
            if (!adapted_) {
                adapted_=true;
                if (diagnostics_) ++diagnostics_->adaptiveMajorantFlights;
            }
            pending_.push_back({middle,interval.end,interval.maximum,true});
            interval.end=middle;
            interval.maximum=std::min(interval.maximum,
                                      bound(interval.begin,interval.end));
        }
        age_=interval.end;
        return ExtinctionSegment{interval.begin,interval.end,interval.maximum,false};
    }
    return std::nullopt;
}

FlightSample ConditionalMedium::sample(const ConditionalFlightKernel& flight,
    Random& rng, DdaTrackingDiagnostics* diagnostics, double maximumAge) const {
    const double end=maximumAge < 0.0 ? flight.maximumAgeInDomain() :
        std::min(maximumAge,flight.maximumAgeInDomain());
    ConditionalMajorantCursor cursor(flight,end,diagnostics);
    return sampleSegmentedDeltaTracking(flight,cursor,rng,end,diagnostics);
}

} // namespace mf
