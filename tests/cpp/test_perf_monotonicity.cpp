#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef STREAMING_CECH_PERF_PATH
#error "STREAMING_CECH_PERF_PATH must be defined with the full path to test_streaming_cech_perf binary"
#endif

// Stronger assertion test: for fixed n,d,radius,maxDim, increasing K should not increase edges/simplices beyond baseline.
// It should be monotonic in K: edges_K1 <= edges_K2 <= edges_baseline and same for simplices.
int main() {
    const char* artifactsEnv = std::getenv("TDA_ARTIFACT_DIR");
    std::string dir = artifactsEnv && std::strlen(artifactsEnv) > 0 ? artifactsEnv : std::string("/tmp");
    if (!dir.empty() && dir.back() == '/') dir.pop_back();

    auto run_case = [&](int K, const char* outName, const char* baseOutName, size_t &edges, size_t &simplices) {
        std::string curJson = dir + "/" + outName;
        std::string baseJson = dir + "/" + baseOutName;
        std::remove(curJson.c_str());
        std::remove(baseJson.c_str());
        std::ostringstream cmd;
        cmd << STREAMING_CECH_PERF_PATH
            << " --n 6000 --d 6 --radius 0.9 --maxDim 2 --maxNeighbors 64 --block 64"
            << " --soft-knn-cap " << K << " --parallel-threshold 0"
            << " --baseline-compare 1 --baseline-separate-process 1"
            << " --baseline-maxDim 2"
            << " --json " << curJson
            << " --baseline-json-out " << baseJson;
        int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            std::cerr << "Harness run failed rc=" << rc << "\n";
            return false;
        }
        // Read last line of each JSONL
        auto read_last = [&](const std::string &path){
            std::ifstream in(path);
            std::string line, last;
            while (std::getline(in, line)) if (!line.empty()) last = line;
            return last;
        };
        std::string cur = read_last(curJson);
        std::string base = read_last(baseJson);
        auto grab = [](const std::string &s, const char* key){
            size_t p = s.find(key);
            if (p == std::string::npos) return (size_t)0;
            p += std::strlen(key);
            return (size_t) std::strtoull(s.c_str() + p, nullptr, 10);
        };
        edges = grab(cur, "\"dm_edges\":");
        simplices = grab(cur, "\"simplices\":");
        size_t edges_base = grab(base, "\"dm_edges\":");
        size_t simplices_base = grab(base, "\"simplices\":");
        // Sanity
        assert(edges > 0 || simplices > 0);
        // Edges/simplices should be <= baseline
        assert(edges <= edges_base);
        assert(simplices <= simplices_base);
        return true;
    };

    size_t e8=0,s8=0,e16=0,s16=0;
    bool ok8 = run_case(8,  "mono_k8.jsonl",  "mono_base_k8.jsonl",  e8,  s8);
    bool ok16= run_case(16, "mono_k16.jsonl", "mono_base_k16.jsonl", e16, s16);
    if (!ok8 || !ok16) return 1;
    // Monotonicity: K=8 <= K=16 (edges and simplices should not decrease when K increases)
    assert(e8 <= e16);
    assert(s8 <= s16);
    std::cout << "Monotonicity test passed: e8=" << e8 << " e16=" << e16
              << " s8=" << s8 << " s16=" << s16 << "\n";
    return 0;
}
