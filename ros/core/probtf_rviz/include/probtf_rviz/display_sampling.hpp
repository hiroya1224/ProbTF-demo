#pragma once

#include <cstdint>
#include <map>
#include <random>
#include <string>

namespace probtf_rviz {

// The seed initializes each physical edge's stream once. New distributions
// consume new draws; only children in the SAME render share an edge sample.
class DisplaySamplingStreams {
 public:
  std::mt19937& generator(uint32_t seed, const std::string& key) {
    if (!initialized_ || seed != seed_) {
      streams_.clear();
      seed_ = seed;
      initialized_ = true;
    }
    auto found = streams_.find(key);
    if (found == streams_.end()) {
      uint64_t hash = 1469598103934665603ULL;
      for (const unsigned char character : key) {
        hash ^= character;
        hash *= 1099511628211ULL;
      }
      std::seed_seq seeds{seed, static_cast<uint32_t>(hash),
                          static_cast<uint32_t>(hash >> 32U)};
      found = streams_.emplace(key, std::mt19937(seeds)).first;
    }
    return found->second;
  }

  void reset() {
    streams_.clear();
    initialized_ = false;
  }

 private:
  bool initialized_ = false;
  uint32_t seed_ = 0;
  std::map<std::string, std::mt19937> streams_;
};

}  // namespace probtf_rviz
