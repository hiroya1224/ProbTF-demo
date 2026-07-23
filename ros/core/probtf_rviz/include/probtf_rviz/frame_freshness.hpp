#pragma once

#include <ros/time.h>

#include <algorithm>
#include <cmath>

namespace probtf_rviz {

struct FreshnessVisualState {
  bool visible = true;
  float alpha = 1.0F;
};

// Tracks progress in a source timestamp without treating repeated snapshots as
// new data.  This is important for /probtf_batch, whose fresh array header can
// contain old, unchanged transform records.
class StampFreshness {
 public:
  void reset() {
    initialized_ = false;
    clock_initialized_ = false;
    source_stamp_ = ros::Time();
    last_progress_ = ros::Time();
    last_clock_ = ros::Time();
  }

  void observe(const ros::Time& source_stamp, const ros::Time& now) {
    rebaseOnClockRewind(now);
    if (!initialized_ || source_stamp != source_stamp_) {
      source_stamp_ = source_stamp;
      last_progress_ = now;
      initialized_ = true;
    }
  }

  void markProgress(const ros::Time& now) {
    rebaseOnClockRewind(now);
    last_progress_ = now;
    initialized_ = true;
  }

  FreshnessVisualState state(const ros::Time& now, double timeout_seconds) {
    rebaseOnClockRewind(now);
    FreshnessVisualState output;
    if (!initialized_ || !std::isfinite(timeout_seconds) ||
        timeout_seconds <= 0.0) {
      return output;
    }

    const double age =
        std::max(0.0, (now - last_progress_).toSec());
    output.visible = age <= timeout_seconds;

    const double third = timeout_seconds / 3.0;
    if (age > 2.0 * third) {
      output.alpha = static_cast<float>(
          std::max(0.0, (timeout_seconds - age) / third));
    }
    return output;
  }

  bool initialized() const {
    return initialized_;
  }

  const ros::Time& sourceStamp() const {
    return source_stamp_;
  }

  const ros::Time& lastProgress() const {
    return last_progress_;
  }

 private:
  void rebaseOnClockRewind(const ros::Time& now) {
    if (clock_initialized_ && now < last_clock_ && initialized_) {
      last_progress_ = now;
    }
    last_clock_ = now;
    clock_initialized_ = true;
  }

  bool initialized_ = false;
  bool clock_initialized_ = false;
  ros::Time source_stamp_;
  ros::Time last_progress_;
  ros::Time last_clock_;
};

}  // namespace probtf_rviz
