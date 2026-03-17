// Copyright 2025, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "base/pmr/memory_resource.h"

namespace dfly {

//
// TOPK: User-Facing API Data Structure
//
// This class implements the data structure required to support the public Redis
// TOPK module API (e.g., TOPK.RESERVE, TOPK.ADD, TOPK.INCRBY).
//
// WHY WE HAVE TWO TOP-K IMPLEMENTATIONS:
// Dragonfly maintains two separate Top-K tracking structures to protect the
// performance of the database's hot path:
// 1. `TopKeys` (src/core/top_keys.h): An internal-only, hyper-optimized O(1)
//    tracker that runs on every single database command to detect hot keys.
//    It intentionally lacks a min-heap and uses standard memory allocation to
//    maximize raw speed and minimize instruction cache pollution.
// 2. `TOPK` (this file): The user-facing implementation. To comply with the Redis
//    API contract, this class MUST support instant eviction reporting (requiring an
//    O(log K) Min-Heap), arbitrary increments, and PMR allocators for strict
//    memory limit tracking and RDB snapshot serialization.
//
// Forcing the internal tracker to support Min-Heaps and PMR would severely
// degrade overall database throughput, hence the strict separation of concerns.
//
// Algorithm Deviation Note:
// While heavily inspired by the HeavyKeeper algorithm, this is NOT a strict
// implementation. The original HeavyKeeper paper requires storing a
// (fingerprint, count) pair in each cell so that decay only penalizes a specific
// item. This implementation uses a bare `uint32_t` counter grid, making it closer
// to a Count-Min Sketch coupled with a Min-Heap and a decay heuristic. This
// design safely overestimates counts (which is acceptable for Top-K bounds)
// while simplifying PMR memory layout and RDB serialization.
//
class TOPK {
 public:
  // Initializes a Top-K tracking sketch with the specified dimensions.
  // k: Maximum number of most frequent items to maintain in the min-heap.
  // width: Number of counter buckets per row in the hash grid.
  // depth: Number of independent hash functions (rows) used.
  // decay: Probability multiplier for exponential decay (must be 0.0 to 1.0).
  TOPK(uint32_t k, uint32_t width, uint32_t depth, double decay,
       PMR_NS::memory_resource* mr = nullptr);
  TOPK(const TOPK&) = delete;
  TOPK& operator=(const TOPK&) = delete;
  TOPK(TOPK&& other) noexcept;
  TOPK& operator=(TOPK&& other) noexcept;
  ~TOPK() = default;

  // Size is 4097 so that (kDecayLookupSize - 1) equals exactly 4096 (2^12).
  // This allows the C++ compiler to optimize the division and modulo operations
  // in the extrapolation hot-path into very-fast bitwise shifts & ANDs.
  static constexpr size_t kDecayLookupSize = 4097;

  static constexpr double kDefaultDecay = 0.9;
  static constexpr double kDecayEpsilon = 1e-9;

  // Represents an item in the Top-K list with its estimated count
  struct TopKItem {
    std::string item;
    uint32_t count;
  };

  // Add an item to the sketch.
  // Returns the evicted item if one was removed from Top-K, or std::nullopt.
  std::optional<std::string> Add(std::string_view item);

  // Add multiple items to the sketch.
  // Returns a vector where each element is either an evicted item or std::nullopt.
  std::vector<std::optional<std::string>> AddMultiple(const std::vector<std::string_view>& items);

  // Increment an item's count by the specified amount.
  // Returns the evicted item if one was removed, or std::nullopt.
  // increment must be > 0.
  std::optional<std::string> IncrBy(std::string_view item, uint32_t increment);

  // Increment multiple items by specified amounts.
  // Returns a vector where each element is either an evicted item or std::nullopt.
  std::vector<std::optional<std::string>> IncrByMultiple(
      const std::vector<std::pair<std::string_view, uint32_t>>& items);

  // Query if items are in the Top-K list.
  // Returns 1 if item is in Top-K, 0 otherwise.
  [[nodiscard]] std::vector<int> Query(const std::vector<std::string_view>& items) const;

  // Get estimated counts for items.
  [[nodiscard]] std::vector<uint32_t> Count(const std::vector<std::string_view>& items) const;

  // Get the Top-K items list, optionally with counts.
  // Items are sorted by estimated frequency (highest first).
  [[nodiscard]] std::vector<TopKItem> List() const;

  // Accessors for Top-K parameters
  [[nodiscard]] uint32_t K() const {
    return k_;
  }

  [[nodiscard]] uint32_t Width() const {
    return width_;
  }

  [[nodiscard]] uint32_t Depth() const {
    return depth_;
  }

  [[nodiscard]] double Decay() const {
    return decay_;
  }

  // Memory usage in bytes
  [[nodiscard]] size_t MallocUsed() const;

  // Serialization support for RDB persistence
  struct SerializedData {
    uint32_t k;
    uint32_t width;
    uint32_t depth;
    double decay;
    std::vector<TopKItem> heap_items;
    std::vector<uint32_t> counters;
  };
  [[nodiscard]] SerializedData Serialize() const;
  void Deserialize(const SerializedData& data);

 private:
  struct HeapItem {
    std::string key;
    uint32_t count;
    size_t hash;  // Pre-computed hash

    // Min heap comparator
    bool operator>(const HeapItem& other) const {
      return count > other.count;
    }
  };

  // Hash function for bucket selection in row
  [[nodiscard]] uint64_t Hash(std::string_view item, uint32_t row) const;

  // Exponential decay logic
  [[nodiscard]] bool ShouldDecay(uint32_t current_count) const;

  // Get the minimum count for an item across all hash table rows
  [[nodiscard]] uint32_t GetMinCount(std::string_view item) const;

  // Updates the min-heap with the new count for the given item.
  // Returns the evicted item's key if the heap is at capacity and a new item displaces an existing
  // one. Otherwise, returns std::nullopt.
  std::optional<std::string> UpdateHeap(std::string_view item, uint32_t new_count);

  // Tries to evict the item with the minimum count (lowest frequency) from the heap
  // if capacity (k_) is reached. Returns the evicted item's key, or std::nullopt.
  std::optional<std::string> TryEvictMin();

  // Check if an item is in the Top-K heap
  [[nodiscard]] bool IsInHeap(std::string_view item) const {
    return item_to_hash_.contains(std::string(item));
  }

  // Hashes the item for a specific row and calculates its flattened 1D index
  // within the counters_ array. Maps the 2D Count-Min Sketch grid (depth x width)
  // into a single contiguous block of memory for better CPU cache locality.
  size_t GetCounterIndex(std::string_view item, uint32_t row) const;

  // Shared increment logic
  std::optional<std::string> IncrementInternal(std::string_view item, uint32_t increment);

  // Compute decay probability using lookup table or extrapolation
  double ComputeDecayProbability(uint32_t count) const;

  // Heap maintenance functions
  // O(log k) ops
  void HeapifyUp(size_t index);
  void HeapifyDown(size_t index);

  uint32_t k_;      // Number of top items to track
  uint32_t width_;  // Hash table width (buckets per row)
  uint32_t depth_;  // Hash table depth (number of rows)
  double decay_;    // Decay constant (0.0-1.0, typically 0.9)

  // Pointer to the active decay lookup table. For the default decay (0.9), this points to
  // a process-wide shared static table (32KB, allocated once). For custom (non-default) decay
  // values, it points to custom_decay_table_ below. This pattern can help to avoid embedding a 32KB
  // array in every TOPK object.
  // Assumption: >99% of TOPK instances will use the default decay, so
  // this optimization can significantly reduce memory usage and improve startup performance by
  // avoiding the need to build a custom table for each instance.
  const std::array<double, kDecayLookupSize>* decay_lookup_ = nullptr;

  // Heap-allocated table for non-default decay values. Null for the common case (decay=0.9).
  std::unique_ptr<std::array<double, kDecayLookupSize>> custom_decay_table_;

  // HeavyKeeper data structures
  // Hash table: width × depth matrix of counters
  std::vector<uint32_t, PMR_NS::polymorphic_allocator<uint32_t>> counters_;

  // Min heap: vector of top-K items maintained as a min heap
  std::vector<HeapItem, PMR_NS::polymorphic_allocator<HeapItem>> min_heap_;

  // O(1) fast-path membership index for the min-heap.
  // Maps items currently in the Top-K list to their pre-computed 64-bit hashes.
  // This serves two critical performance goals:
  // 1. Prevents O(K) linear scans just to check if an item is currently in the heap.
  // 2. Caches the hash to allow very fast integer comparisons instead of
  //    slow string comparisons when locating the item inside the heap array.
  absl::flat_hash_map<std::string, size_t> item_to_hash_;
};

}  // namespace dfly
