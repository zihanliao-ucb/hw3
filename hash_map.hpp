// #pragma once

// #include "kmer_t.hpp"
// #include <upcxx/upcxx.hpp>
// #include <vector>
// #include <list>

// struct HashMap {
//     // Distributed object containing the local hash table with separate chaining
//     upcxx::dist_object<std::vector<std::list<kmer_pair>>> table;
//     size_t bucket_count;

//     HashMap(size_t buckets);

//     size_t size() const noexcept;

//     // Insert and find operations
//     upcxx::future<bool> insert(const kmer_pair& kmer);
//     upcxx::future<std::optional<kmer_pair>> find(const pkmer_t& key_kmer);

//     int get_target_rank(const pkmer_t& key_kmer) const;
//     size_t get_bucket_index(const pkmer_t& key_kmer) const;
// };

// HashMap::HashMap(size_t buckets)
//     : table(std::vector<std::list<kmer_pair>>(buckets)),
//       bucket_count(buckets) {}

// size_t HashMap::size() const noexcept {
//     return bucket_count;
// }

// int HashMap::get_target_rank(const pkmer_t& key_kmer) const {
//     return key_kmer.hash() % upcxx::rank_n();
// }

// size_t HashMap::get_bucket_index(const pkmer_t& key_kmer) const {
//     return key_kmer.hash() % bucket_count;
// }

// upcxx::future<bool> HashMap::insert(const kmer_pair& kmer) {
//     int target_rank = get_target_rank(kmer.kmer);
//     return upcxx::rpc(target_rank,
//                       [](upcxx::dist_object<std::vector<std::list<kmer_pair>>>& table,
//                          size_t bucket_count,
//                          const kmer_pair& kmer) -> bool {
//                           size_t bucket_index = kmer.kmer.hash() % bucket_count;
//                           auto& bucket = (*table)[bucket_index];
//                           // Check if the kmer already exists
//                           for (const auto& entry : bucket) {
//                               if (entry.kmer == kmer.kmer) {
//                                   return false; // kmer already exists
//                               }
//                           }
//                           bucket.push_back(kmer);
//                           return true;
//                       }, table, bucket_count, kmer);
// }

// upcxx::future<std::optional<kmer_pair>> HashMap::find(const pkmer_t& key_kmer) {
//     int target_rank = get_target_rank(key_kmer);
//     return upcxx::rpc(target_rank,
//                       [](upcxx::dist_object<std::vector<std::list<kmer_pair>>>& table,
//                          size_t bucket_count,
//                          const pkmer_t& key_kmer) -> std::optional<kmer_pair> {
//                           size_t bucket_index = key_kmer.hash() % bucket_count;
//                           const auto& bucket = (*table)[bucket_index];
//                           for (const auto& entry : bucket) {
//                               if (entry.kmer == key_kmer) {
//                                   return entry; // kmer found
//                               }
//                           }
//                           return std::nullopt; // kmer not found
//                       }, table, bucket_count, key_kmer);
// }

#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>
#include <vector>
#include <optional>

struct Ext {
    char backwardExt;
    char forwardExt;
};

struct HashMap {
    // Each rank holds a local linear-probing-based hash table
    upcxx::dist_object<std::vector<std::optional<kmer_pair>>> table;
    size_t bucket_count;

    HashMap(size_t buckets);

    size_t size() const noexcept;

    // Insert and find operations
    upcxx::future<bool> insert(const kmer_pair& kmer);
    upcxx::future<bool> insert(const std::vector<kmer_pair>& kmers);
    upcxx::future<std::optional<kmer_pair>> find(const pkmer_t& key_kmer);
    upcxx::future<std::vector<std::optional<Ext>>> find(const std::vector<pkmer_t>& key_kmers);

    int get_target_rank(const pkmer_t& key_kmer) const;
    size_t get_bucket_index(const pkmer_t& key_kmer) const;
};

HashMap::HashMap(size_t buckets)
    : table(std::vector<std::optional<kmer_pair>>(buckets)), bucket_count(buckets) {}

size_t HashMap::size() const noexcept {
    return bucket_count;
}

int HashMap::get_target_rank(const pkmer_t& key_kmer) const {
    return key_kmer.hash() % upcxx::rank_n();
}

size_t HashMap::get_bucket_index(const pkmer_t& key_kmer) const {
    return key_kmer.hash() % bucket_count;
}

upcxx::future<bool> HashMap::insert(const kmer_pair& kmer) {
    int target_rank = get_target_rank(kmer.kmer);
    return upcxx::rpc(target_rank,
                      [](upcxx::dist_object<std::vector<std::optional<kmer_pair>>>& table,
                         size_t bucket_count,
                         const kmer_pair& kmer) -> bool {
                          size_t idx = kmer.kmer.hash() % bucket_count;
                          for (size_t i = 0; i < bucket_count; ++i) {
                              size_t probe_idx = (idx + i) % bucket_count;
                              auto& slot = (*table)[probe_idx];

                              if (!slot.has_value()) {
                                  slot = kmer;
                                  return true;
                              } else if (slot->kmer == kmer.kmer) {
                                  return false; // already exists
                              }
                          }
                          return false; // table full
                      }, table, bucket_count, kmer);
}

upcxx::future<bool> HashMap::insert(const std::vector<kmer_pair>& kmers) {
    if (kmers.empty()) {
        return upcxx::make_future(true);
    }

    int target_rank = get_target_rank(kmers[0].kmer); // assume all kmers go to same rank
    return upcxx::rpc(target_rank,
                      [](upcxx::dist_object<std::vector<std::optional<kmer_pair>>>& table,
                         size_t bucket_count,
                         const std::vector<kmer_pair>& kmers) -> bool {
                          auto& local_table = *table;
                          for (const auto& kmer : kmers) {
                              size_t idx = kmer.kmer.hash() % bucket_count;
                              bool inserted = false;

                              for (size_t i = 0; i < bucket_count; ++i) {
                                  size_t probe_idx = (idx + i) % bucket_count;
                                  auto& slot = local_table[probe_idx];

                                  if (!slot.has_value()) {
                                      slot = kmer;
                                      inserted = true;
                                      break;
                                  } else if (slot->kmer == kmer.kmer) {
                                      inserted = true; // already exists, that's okay
                                      break;
                                  }
                              }

                              if (!inserted) {
                                  return false; // at least one failed due to full table
                              }
                          }
                          return true;
                      }, table, bucket_count, kmers);
}

upcxx::future<std::optional<kmer_pair>> HashMap::find(const pkmer_t& key_kmer) {
    int target_rank = get_target_rank(key_kmer);
    return upcxx::rpc(target_rank,
                      [](upcxx::dist_object<std::vector<std::optional<kmer_pair>>>& table,
                         size_t bucket_count,
                         const pkmer_t& key_kmer) -> std::optional<kmer_pair> {
                          size_t idx = key_kmer.hash() % bucket_count;
                          for (size_t i = 0; i < bucket_count; ++i) {
                              size_t probe_idx = (idx + i) % bucket_count;
                              const auto& slot = (*table)[probe_idx];

                              if (!slot.has_value()) {
                                  return std::nullopt; // not found
                              } else if (slot->kmer == key_kmer) {
                                  return slot;
                              }
                          }
                          return std::nullopt; // not found
                      }, table, bucket_count, key_kmer);
}

upcxx::future<std::vector<std::optional<Ext>>> HashMap::find(const std::vector<pkmer_t>& key_kmers) {
    if (key_kmers.empty()) {
        return upcxx::make_future(std::vector<std::optional<Ext>>{});
    }

    int target_rank = get_target_rank(key_kmers[0]); // assumes all keys go to same rank
    return upcxx::rpc(target_rank,
                      [](upcxx::dist_object<std::vector<std::optional<kmer_pair>>>& table,
                         size_t bucket_count,
                         const std::vector<pkmer_t>& key_kmers) -> std::vector<std::optional<Ext>> {
                          const auto& local_table = *table;
                          std::vector<std::optional<Ext>> results;
                          results.reserve(key_kmers.size());

                          for (const auto& key : key_kmers) {
                              size_t idx = key.hash() % bucket_count;
                              std::optional<Ext> result = std::nullopt;

                              for (size_t i = 0; i < bucket_count; ++i) {
                                  size_t probe_idx = (idx + i) % bucket_count;
                                  const auto& slot = local_table[probe_idx];

                                  if (!slot.has_value()) {
                                      break;
                                  } else if (slot->kmer == key) {
                                      result = Ext{slot->fb_ext[0], slot->fb_ext[1]};
                                      break;
                                  }
                              }

                              results.push_back(result);
                          }
                          return results;
                      }, table, bucket_count, key_kmers);
}
