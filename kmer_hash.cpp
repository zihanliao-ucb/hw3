#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <numeric>
#include <set>
#include <optional>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "hash_map.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"

#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    std::string test_prefix = "test";
    if (run_type == "test" && argc >= 4) {
        test_prefix = std::string(argv[3]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers;
    size_t local_size;

    if (upcxx::rank_me() == 0) {
        n_kmers = line_count(kmer_fname);
        // Compute hash_table_size with a load factor of 0.5
        size_t hash_table_size = n_kmers * 2;
        // Calculate local_size on the root process
        local_size = (hash_table_size + upcxx::rank_n() - 1) / upcxx::rank_n();
    }

    n_kmers = upcxx::broadcast(n_kmers, 0).wait();
    local_size = upcxx::broadcast(local_size, 0).wait();

    HashMap hashmap(local_size);

    if (run_type == "verbose") {
        BUtil::print("Initializing hash table of size %d for %d kmers.\n", local_size,
                     n_kmers);
    }

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<upcxx::future<bool>> insert_futures;
    std::vector<kmer_pair> start_nodes;
    std::vector<std::vector<kmer_pair>> inserts(upcxx::rank_n(), std::vector<kmer_pair>());
    
    for (auto& kmer : kmers) {
        int target_rank = hashmap.get_target_rank(kmer.kmer);
        inserts[target_rank].push_back(kmer);
        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }

    constexpr size_t chunk_size = 100;
    
    for (int i = 0; i < upcxx::rank_n(); i++) {
        const auto& batch = inserts[i];
        size_t total = batch.size();
        if (total == 0) continue;
    
        for (size_t offset = 0; offset < total; offset += chunk_size) {
            size_t end = std::min(offset + chunk_size, total);
            std::vector<kmer_pair> chunk(batch.begin() + offset, batch.begin() + end);
    
            insert_futures.push_back(
                hashmap.insert(chunk).then(
                    [i](bool success) {
                        if (!success) {
                            throw std::runtime_error("Error: HashMap insertion failed.");
                        }
                        return success;
                    }
                )
            );
        }
    }

    upcxx::when_all(insert_futures.begin(), insert_futures.end()).wait();
    auto end_insert = std::chrono::high_resolution_clock::now();
    upcxx::barrier();

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Finished inserting in %lf\n", insert_time);
    }
    upcxx::barrier();

    auto start_read = std::chrono::high_resolution_clock::now();

    std::list<std::list<kmer_pair>> contigs;

    // for (const auto& start_kmer : start_nodes) {
    //     std::list<kmer_pair> contig;
    //     contig.push_back(start_kmer);
    
    //     while (contig.back().forwardExt() != 'F') {
    //         pkmer_t next_kmer = contig.back().next_kmer();
    //         kmer_pair kmer;
    //         bool found = hashmap.find(next_kmer).then(
    //             [&kmer](std::optional<kmer_pair> result) -> bool {
    //                 if (result.has_value()) {
    //                     kmer = result.value();
    //                     return true;
    //                 } else {
    //                     return false;
    //                 }
    //             }
    //         ).wait();
    
    //         if (!found) {
    //             throw std::runtime_error("Error: k-mer not found in hashmap.");
    //         }
    
    //         contig.push_back(kmer);
    //     }
    
    //     contigs.push_back(contig);
    // }    

    std::list<std::list<kmer_pair>> ongoing_contigs;
    for (kmer_pair& kmer : start_nodes) {
        std::list<kmer_pair> contig;
        contig.push_back(kmer);
        if (kmer.forwardExt() == 'F') {
            contigs.push_back(contig);
        } else {
            ongoing_contigs.push_back(contig);
        }
    }

    while (!ongoing_contigs.empty()) {
        std::vector<std::vector<pkmer_t>> finds(upcxx::rank_n());
        std::vector<std::vector<std::list<std::list<kmer_pair>>::iterator>> contig_iters(upcxx::rank_n());
        std::vector<std::vector<upcxx::future<std::vector<std::optional<kmer_pair>>>>> find_futures(upcxx::rank_n());
    
        // Step 1: Build queries from last kmer of each ongoing contig
        // std::cerr << upcxx::rank_me() << ": [DEBUG] Building queries from last kmer of each ongoing contig\n";
        for (auto it = ongoing_contigs.begin(); it != ongoing_contigs.end(); ++it) {
            kmer_pair& tail = it->back();
            int rank = hashmap.get_target_rank(tail.next_kmer());
            finds[rank].push_back(tail.next_kmer());
            contig_iters[rank].push_back(it); // keep track of which contig each query belongs to
        }
    
        // Step 2: Chunked batched find across ranks
        // std::cerr << upcxx::rank_me() << ": [DEBUG] Chunked batched find across ranks\n";
        for (int rank = 0; rank < upcxx::rank_n(); ++rank) {
            const auto& keys = finds[rank];
            size_t total = keys.size();
            if (total == 0) continue;
    
            for (size_t offset = 0; offset < total; offset += chunk_size) {
                size_t end = std::min(offset + chunk_size, total);
                std::vector<pkmer_t> chunk(keys.begin() + offset, keys.begin() + end);
    
                find_futures[rank].push_back(hashmap.find(chunk));
            }
        }
    
        // Step 3: Wait for all results
        // std::cerr << upcxx::rank_me() << ": [DEBUG] Finished waiting for all results\n";
        std::vector<std::vector<std::optional<kmer_pair>>> results_batches;
        for (int rank = 0; rank < upcxx::rank_n(); ++rank) {
            std::vector<std::optional<kmer_pair>> results;
            for (const auto& fut : find_futures[rank]) {
                const auto& result = fut.wait();
                results.insert(results.end(), result.begin(), result.end());
            }
            results_batches.push_back(results);
        }
    
        // Step 4: Process results and update contigs
        // std::cerr << upcxx::rank_me() << ": [DEBUG] Processing results and updating contigs\n";
        for (int rank = 0; rank < upcxx::rank_n(); ++rank) {
            for (size_t i = 0; i < results_batches[rank].size(); ++i) {
                const auto& opt_kmer = results_batches[rank][i];
                auto contig_it = contig_iters[rank][i];
        
                if (opt_kmer.has_value()) {
                    contig_it->push_back(opt_kmer.value());
                    // std::cerr << upcxx::rank_me() << ": [DEBUG] Completed contig." << std::endl;
                    if (opt_kmer.value().forwardExt() == 'F') {
                        // std::cerr << upcxx::rank_me() << ": [DEBUG] Completed contig." << std::endl;
                        contigs.push_back(std::move(*contig_it));
                        ongoing_contigs.erase(contig_it);
                    }
                } else {
                    throw std::runtime_error("Error: k-mer not found in hashmap.");
                }
            }
        }
    }
    
    auto end_read = std::chrono::high_resolution_clock::now();
    upcxx::barrier();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> read = end_read - start_read;
    std::chrono::duration<double> insert = end_insert - start;
    std::chrono::duration<double> total = end - start;

    int numKmers = std::accumulate(
        contigs.begin(), contigs.end(), 0,
        [](int sum, const std::list<kmer_pair>& contig) { return sum + contig.size(); });

    if (run_type != "test") {
        BUtil::print("Assembled in %lf total\n", total.count());
    }

    if (run_type == "verbose") {
        printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
               " (%lf read, %lf insert, %lf total)\n",
               upcxx::rank_me(), contigs.size(), numKmers, start_nodes.size(), read.count(),
               insert.count(), total.count());
    }

    if (run_type == "test") {
        std::ofstream fout(test_prefix + "_" + std::to_string(upcxx::rank_me()) + ".dat");
        for (const auto& contig : contigs) {
            fout << extract_contig(contig) << std::endl;
        }
        fout.close();
    }

    upcxx::finalize();
    return 0;
}
