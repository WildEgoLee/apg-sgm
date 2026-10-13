#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/pipeline_stats.hpp"
#include "apg_sgm/session.hpp"

#include <iostream>
#include <cassert>
#include <string>

using namespace apg;

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "               TEST OBSERVABILITY, DIAGNOSTICS & RESOURCE ACCOUNTING      " << std::endl;
    std::cout << "==========================================================================" << std::endl;

    const int W = 32;
    const int H = 32;
    Image8 left(W, H, 1);
    Image8 right(W, H, 1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            left.at(x, y) = static_cast<uint8_t>((x * 17 + y * 23) % 256);
            int xr = (x >= 4) ? (x - 4) : 0;
            right.at(x, y) = static_cast<uint8_t>((xr * 17 + y * 23) % 256);
        }
    }

    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Fast);
    cfg.min_disparity = 0;
    cfg.max_disparity = 16;
    StereoMatcher matcher(cfg);

    // 1. Diagnostics with ReleasePerFrame
    PipelineStats stats_release;
    {
        StereoResult res;
        StereoWorkspace ws; // ReleasePerFrame
        Status s = matcher.compute_status(left, right, res, ws, &stats_release);
        if (!s.ok()) {
            std::cerr << "compute_status failed: " << s.message << "\n";
            return 1;
        }

        if (stats_release.timing.total_ms <= 0.0) {
            std::cerr << "Expected positive total_ms timing\n";
            return 1;
        }
        if (stats_release.timing.aux_ms <= 0.0) {
            std::cerr << "Expected positive aux_ms timing\n";
            return 1;
        }
        if (stats_release.peak_workspace_bytes == 0) {
            std::cerr << "Expected non-zero peak_workspace_bytes\n";
            return 1;
        }
        if (stats_release.retained_workspace_bytes != 0) {
            std::cerr << "Expected 0 retained_workspace_bytes under ReleasePerFrame, got "
                      << stats_release.retained_workspace_bytes << "\n";
            return 1;
        }
        if (stats_release.valid_pixels == 0 || stats_release.valid_disparity_ratio <= 0.0f) {
            std::cerr << "Expected valid disparity pixels\n";
            return 1;
        }
        std::cout << "  ReleasePerFrame diagnostics and resource accounting: PASS\n";
    }

    // 2. Diagnostics with BoundedReuse
    PipelineStats stats_reuse;
    {
        StereoResult res;
        ExecutionOptions opt;
        opt.memory_policy = MemoryPolicy::BoundedReuse;
        StereoWorkspace ws(opt);

        Status s = matcher.compute_status(left, right, res, ws, &stats_reuse);
        if (!s.ok()) {
            std::cerr << "compute_status reuse failed: " << s.message << "\n";
            return 1;
        }

        if (stats_reuse.retained_workspace_bytes == 0) {
            std::cerr << "Expected non-zero retained_workspace_bytes under BoundedReuse\n";
            return 1;
        }
        if (stats_reuse.retained_workspace_bytes != ws.allocated_bytes()) {
            std::cerr << "retained_workspace_bytes does not match ws.allocated_bytes()\n";
            return 1;
        }
        std::cout << "  BoundedReuse memory retention tracking: PASS\n";
    }

    // 3. String & JSON serialization formatting
    {
        std::string report = stats_release.to_string();
        if (report.find("=== Pipeline Diagnostics & Stats ===") == std::string::npos ||
            report.find("Peak Workspace:") == std::string::npos ||
            report.find("Valid Ratio:") == std::string::npos) {
            std::cerr << "to_string() missing key diagnostics markers\n";
            return 1;
        }

        std::string json = stats_release.to_json();
        if (json.front() != '{' || json.back() != '}' ||
            json.find("\"timing\"") == std::string::npos ||
            json.find("\"memory\"") == std::string::npos ||
            json.find("\"peak_workspace_bytes\"") == std::string::npos ||
            json.find("\"valid_disparity_ratio\"") == std::string::npos) {
            std::cerr << "to_json() missing expected JSON structure\n";
            return 1;
        }
        std::cout << "  Diagnostic serialization (to_string, to_json): PASS\n";
    }

    // 4. Instrumentation via StereoSession
    {
        StereoSession session(cfg);
        StereoResult res;
        PipelineStats session_stats;
        Status s = session.process_status(left, right, res, &session_stats);
        if (!s.ok()) {
            std::cerr << "Session process_status failed: " << s.message << "\n";
            return 1;
        }
        if (session_stats.timing.total_ms <= 0.0) {
            std::cerr << "Session stats missing timing\n";
            return 1;
        }
        std::cout << "  StereoSession instrumentation: PASS\n";
    }

    std::cout << "All Observability & Diagnostics tests passed!\n";
    return 0;
}
