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

int main() {
    const char* artifactsEnv = std::getenv("TDA_ARTIFACT_DIR");
    std::string dir = artifactsEnv && std::strlen(artifactsEnv) > 0 ? artifactsEnv : std::string("/tmp");
    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    std::string outPath = dir + "/manifest_presence.jsonl";
    std::remove(outPath.c_str());

    // Run DM-only quick probe with time-limit to trigger early-stop fields
    std::ostringstream cmd;
    cmd << STREAMING_CECH_PERF_PATH
        << " --dm-only --n 3000 --d 3 --radius 0.8 --block 64"
        << " --time-limit 0.001 --parallel-threshold 1"
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
    auto has = [&](const char* key){ return last.find(key) != std::string::npos; };
    assert(has("\"manifest_hash\""));
    assert(has("\"attempted_count\""));
    assert(has("\"dm_stopped_by_time\""));
    assert(has("\"dm_last_block_i0\""));
    std::cout << "Manifest and early-stop fields present\n";
    return 0;
}




