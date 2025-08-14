#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef STREAMING_CECH_PERF_PATH
#error "STREAMING_CECH_PERF_PATH must be defined with the full path to test_streaming_cech_perf binary"
#endif

// Very small smoke test: run perf harness with separate-process baseline enabled and JSON outputs,
// then parse the JSONL files and assert that expected keys exist and numeric fields are sane.
int main() {
    // Determine artifacts directory (CI-friendly). Use env var TDA_ARTIFACT_DIR if set, else /tmp.
    const char* artifactsEnv = std::getenv("TDA_ARTIFACT_DIR");
    std::string artifactsDir = artifactsEnv && std::strlen(artifactsEnv) > 0 ? artifactsEnv : std::string("/tmp");
    if (!artifactsDir.empty() && artifactsDir.back() == '/') artifactsDir.pop_back();
    std::string parentPath = artifactsDir + "/scp_parent.jsonl";
    std::string baselinePath = artifactsDir + "/scp_baseline.jsonl";

    const char* tmp1 = parentPath.c_str();
    const char* tmp2 = baselinePath.c_str();

    // Clean up any previous leftovers
    std::remove(tmp1);
    std::remove(tmp2);

    // Build command: small n for speed; ensure baseline compare + separate process; write JSONL
    std::ostringstream cmd;
    cmd << STREAMING_CECH_PERF_PATH
        << " --n 800 --d 3 --radius 0.9 --maxDim 1 --maxNeighbors 16 --block 32"
        << " --soft-knn-cap 8 --parallel-threshold 0"
        << " --baseline-compare 1 --baseline-separate-process 1"
        << " --json " << tmp1
        << " --baseline-json-out " << tmp2
        << " --baseline-maxDim 1";

    // Child process baseline JSON will be written to tmp2 via --baseline-json-out.

    int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        std::cerr << "Perf harness command failed: rc=" << rc << "\n";
        return 1;
    }

    // Parse parent JSONL
    std::ifstream fin1(tmp1);
    if (!fin1.good()) {
        std::cerr << "Missing parent JSONL output: " << tmp1 << "\n";
        return 1;
    }
    std::string line;
    bool saw_dm_edges = false, saw_dm_blocks = false, saw_mode = false;
    size_t dm_edges = 0; size_t dm_blocks = 0; std::string mode;
    while (std::getline(fin1, line)) {
        if (line.find("\"dm_edges\"") != std::string::npos) {
            saw_dm_edges = true;
            auto pos = line.find("\"dm_edges\":");
            if (pos != std::string::npos) dm_edges = std::stoull(line.substr(pos + 11));
        }
        if (line.find("\"dm_blocks\"") != std::string::npos) {
            saw_dm_blocks = true;
            auto pos = line.find("\"dm_blocks\":");
            if (pos != std::string::npos) dm_blocks = std::stoull(line.substr(pos + 12));
        }
        if (line.find("\"mode\"") != std::string::npos) {
            saw_mode = true;
        }
    }
    assert(saw_dm_edges && "dm_edges missing in parent JSONL");
    assert(saw_dm_blocks && "dm_blocks missing in parent JSONL");
    assert(saw_mode && "mode missing in parent JSONL");
    assert(dm_blocks > 0);

    // Optionally parse baseline JSONL if present (child process output). Not mandatory.
    std::ifstream fin2(tmp2);
    if (fin2.good()) {
        bool ok = false;
        while (std::getline(fin2, line)) {
            if (line.find("\"dm_edges\"") != std::string::npos) { ok = true; break; }
        }
        assert(ok && "baseline JSONL present but missing dm_edges");
    }

    std::cout << "Perf baseline JSONL test passed. dm_edges=" << dm_edges << ", dm_blocks=" << dm_blocks
              << "; parent_jsonl='" << parentPath << "' baseline_jsonl='" << baselinePath << "'\n";
    return 0;
}
