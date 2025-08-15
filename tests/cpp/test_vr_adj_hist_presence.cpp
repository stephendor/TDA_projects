// Ensure VR mode writes an adjacency histogram CSV for analyzer gating
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
    std::string csv = dir + "/adj_vr_presence_parent.csv";
    std::string json = dir + "/run_vr_presence.jsonl";
    std::remove(csv.c_str());
    std::remove(json.c_str());

    // Small, quick VR run; require CSV output
    std::ostringstream cmd;
    cmd << STREAMING_CECH_PERF_PATH
        << " --mode vr --n 1800 --d 3 --epsilon 0.7 --maxDim 1"
        << " --soft-knn-cap 8 --parallel-threshold 0"
        << " --adj-hist-csv " << csv
        << " --json " << json;
    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "VR harness run failed rc=" << rc << "\n";
        return 1;
    }

    // CSV must exist and be non-empty with header degree,count
    std::ifstream in(csv);
    if (!in.good()) {
        std::cerr << "Missing VR adjacency CSV: " << csv << "\n";
        return 1;
    }
    std::string header;
    std::getline(in, header);
    if (header.find("degree") == std::string::npos || header.find("count") == std::string::npos) {
        std::cerr << "CSV header missing expected columns: " << header << "\n";
        return 1;
    }
    // Ensure at least one data row
    std::string line;
    bool hasRow = false;
    while (std::getline(in, line)) {
        if (!line.empty()) { hasRow = true; break; }
    }
    assert(hasRow);
    std::cout << "VR adjacency CSV present with data: " << csv << "\n";
    return 0;
}


