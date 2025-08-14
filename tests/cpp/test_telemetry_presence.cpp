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

// Focused test: ensure JSONL contains required telemetry fields and that overshoot is zero
// when running with serial threshold and soft cap enabled (dm-only for speed).
int main() {
    const char* artifactsEnv = std::getenv("TDA_ARTIFACT_DIR");
    std::string dir = artifactsEnv && std::strlen(artifactsEnv) > 0 ? artifactsEnv : std::string("/tmp");
    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    std::string outPath = dir + "/telemetry_presence.jsonl";
    std::remove(outPath.c_str());

    // Small, quick run; dm-only; soft cap active; serial threshold
    std::ostringstream cmd;
    cmd << STREAMING_CECH_PERF_PATH
        << " --n 1200 --d 3 --radius 0.9 --block 32 --dm-only"
        << " --soft-knn-cap 8 --parallel-threshold 0"
        << " --json " << outPath;
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "Harness run failed rc=" << rc << "\n";
        return 1;
    }

    std::ifstream in(outPath);
    if (!in.good()) {
        std::cerr << "Missing JSONL output: " << outPath << "\n";
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

    std::cout << "Telemetry presence test passed. pt=" << pt
              << " overshoot_sum=" << oversum << " overshoot_max=" << overmax
              << " jsonl='" << outPath << "'\n";
    return 0;
}
