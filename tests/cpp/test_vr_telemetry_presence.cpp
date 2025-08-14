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

// Ensure JSONL contains required telemetry fields in VR mode and overshoot==0
int main() {
    const char* artifactsEnv = std::getenv("TDA_ARTIFACT_DIR");
    std::string dir = artifactsEnv && std::strlen(artifactsEnv) > 0 ? artifactsEnv : std::string("/tmp");
    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    std::string outPath = dir + "/vr_telemetry_presence.jsonl";
    std::remove(outPath.c_str());

    // Small, quick VR run; soft cap active; serial threshold
    std::ostringstream cmd;
    cmd << STREAMING_CECH_PERF_PATH
        << " --mode vr --n 1500 --d 3 --epsilon 0.8 --maxDim 1"
        << " --soft-knn-cap 8 --parallel-threshold 0"
        << " --json " << outPath;
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "VR harness run failed rc=" << rc << "\n";
        return 1;
    }

    std::ifstream in(outPath);
    if (!in.good()) {
        std::cerr << "Missing VR JSONL output: " << outPath << "\n";
        return 1;
    }
    std::string line, last;
    while (std::getline(in, line)) if (!line.empty()) last = line;
    // Presence checks
    auto has = [&](const char* key){ return last.find(key) != std::string::npos; };
    assert(has("\"dm_peakMB\""));
    assert(has("\"rss_peakMB\""));
    assert(has("\"softcap_overshoot_sum\""));
    assert(has("\"softcap_overshoot_max\""));
    assert(has("\"parallel_threshold\""));
    // Pool telemetry presence (aggregate fields)
    assert(has("\"pool_total_blocks\""));
    assert(has("\"pool_free_blocks\""));
    assert(has("\"pool_fragmentation\""));
    // Optional per-bucket compact string field present
    assert(has("\"pool_bucket_stats_compact\""));
    // Numeric value checks: pt==0 and overshoot==0
    auto grab_int = [&](const char* key){
        size_t p = last.find(key);
        if (p == std::string::npos) return (long long)-1;
        p += std::strlen(key);
        return (long long) std::strtoll(last.c_str() + p, nullptr, 10);
    };
    long long pt = grab_int("\"parallel_threshold\":");
    long long oversum = grab_int("\"softcap_overshoot_sum\":");
    long long overmax = grab_int("\"softcap_overshoot_max\":");
    assert(pt == 0);
    assert(oversum == 0);
    assert(overmax == 0);

    std::cout << "VR telemetry presence test passed. pt=" << pt
              << " overshoot_sum=" << oversum << " overshoot_max=" << overmax
              << " jsonl='" << outPath << "'\n";
    return 0;
}
