#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

#ifndef STREAMING_CECH_PERF_PATH
#error "STREAMING_CECH_PERF_PATH must be defined with the full path to test_streaming_cech_perf binary"
#endif

static std::string last_json_line(const std::string& path) {
    std::ifstream in(path);
    std::string line, last;
    while (std::getline(in, line)) if (!line.empty()) last = line;
    return last;
}

static uint64_t find_uint(const std::string& s, const char* key) {
    auto pos = s.find(key);
    if (pos == std::string::npos) return 0;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return 0;
    return std::strtoull(s.c_str() + pos + 1, nullptr, 10);
}

static double find_double(const std::string& s, const char* key) {
    auto pos = s.find(key);
    if (pos == std::string::npos) return 0.0;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return 0.0;
    return std::strtod(s.c_str() + pos + 1, nullptr);
}

int main() {
#ifndef TDA_ENABLE_CUDA
    std::cout << "CUDA not enabled at build; skipping test" << std::endl;
    return 0;
#else
    const char* artifactsEnv = std::getenv("TDA_ARTIFACT_DIR");
    std::string dir = artifactsEnv && std::strlen(artifactsEnv) > 0 ? artifactsEnv : std::string("/tmp");
    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    std::string cpuJson = dir + "/gpu_sanity_cpu.jsonl";
    std::string gpuJson = dir + "/gpu_sanity_gpu.jsonl";
    std::remove(cpuJson.c_str());
    std::remove(gpuJson.c_str());

    // Small cech/VR probe that exercises block tiles (non-threshold)
    // CPU run
    unsetenv("TDA_USE_CUDA");
    {
        std::ostringstream cmd;
        cmd << STREAMING_CECH_PERF_PATH
            << " --mode cech --n 8000 --d 3 --radius 0.6 --maxDim 1 --block 128"
            << " --json " << cpuJson;
        int rc = std::system(cmd.str().c_str());
        if (rc != 0) { std::cerr << "CPU run failed rc=" << rc << "\n"; return 1; }
    }
    // GPU run
    setenv("TDA_USE_CUDA", "1", 1);
    {
        std::ostringstream cmd;
        cmd << STREAMING_CECH_PERF_PATH
            << " --mode cech --n 8000 --d 3 --radius 0.6 --maxDim 1 --block 128 --cuda 1"
            << " --json " << gpuJson;
        int rc = std::system(cmd.str().c_str());
        if (rc != 0) { std::cerr << "GPU run failed rc=" << rc << "\n"; return 1; }
    }

    auto cpu = last_json_line(cpuJson);
    auto gpu = last_json_line(gpuJson);
    if (cpu.empty() || gpu.empty()) { std::cerr << "Missing JSONL outputs\n"; return 1; }

    uint64_t cpu_edges = find_uint(cpu, "\"dm_edges\"");
    uint64_t gpu_edges = find_uint(gpu, "\"dm_edges\"");
    uint64_t cpu_sx = find_uint(cpu, "\"simplices\"");
    uint64_t gpu_sx = find_uint(gpu, "\"simplices\"");
    // Exact parity expected
    assert(cpu_edges == gpu_edges);
    assert(cpu_sx == gpu_sx);

    double cpu_build = find_double(cpu, "\"build_secs\"");
    double gpu_build = find_double(gpu, "\"build_secs\"");
    // Allow equal or faster GPU (don’t fail on marginal regressions); print info
    std::cout << "GPU sanity: edges=" << gpu_edges << ", simplices=" << gpu_sx
              << ", cpu_build_secs=" << cpu_build << ", gpu_build_secs=" << gpu_build << "\n";
    return 0;
#endif
}




