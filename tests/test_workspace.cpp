#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/workspace.hpp"
#include "apg_sgm/execution.hpp"
#include "apg_sgm/image.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace apg;

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "               TEST WORKSPACE & RESOURCE POLICY ARCHITECTURE              " << std::endl;
    std::cout << "==========================================================================" << std::endl;

    // 1. Workspace default lifecycle & options
    {
        StereoWorkspace ws;
        if (ws.options().memory_policy != MemoryPolicy::ReleasePerFrame) {
            std::cerr << "Expected default MemoryPolicy::ReleasePerFrame\n";
            return 1;
        }
        if (ws.options().max_retained_bytes != 0) {
            std::cerr << "Expected default max_retained_bytes == 0\n";
            return 1;
        }
        if (ws.allocated_bytes() != 0) {
            std::cerr << "Expected initial allocated_bytes == 0\n";
            return 1;
        }
        std::cout << "  Default workspace construction: PASS\n";
    }

    // Synthetic test images
    const int W = 32;
    const int H = 32;
    Image8 left(W, H, 1);
    Image8 right(W, H, 1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            left.at(x, y) = static_cast<uint8_t>((x * 7 + y * 13) % 256);
            int xr = (x >= 4) ? (x - 4) : 0;
            right.at(x, y) = static_cast<uint8_t>((xr * 7 + y * 13) % 256);
        }
    }

    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Fast);
    cfg.min_disparity = 0;
    cfg.max_disparity = 16;
    StereoMatcher matcher(cfg);

    // 2. Test ReleasePerFrame policy
    {
        StereoWorkspace ws; // default ReleasePerFrame
        StereoResult res;
        Status s = matcher.compute_status(left, right, res, ws);
        if (!s.ok()) {
            std::cerr << "compute_status failed: " << s.message << "\n";
            return 1;
        }
        if (res.disparity.width() != W || res.disparity.height() != H) {
            std::cerr << "Disparity dimension mismatch\n";
            return 1;
        }
        if (ws.allocated_bytes() != 0) {
            std::cerr << "Expected ReleasePerFrame to release workspace buffers, but got "
                      << ws.allocated_bytes() << " bytes\n";
            return 1;
        }
        std::cout << "  ReleasePerFrame zero-retention verification: PASS\n";
    }

    // 3. Test BoundedReuse policy & sequential execution
    StereoResult res_reuse1;
    StereoResult res_reuse2;
    {
        ExecutionOptions opt;
        opt.memory_policy = MemoryPolicy::BoundedReuse;
        StereoWorkspace ws(opt);

        Status s1 = matcher.compute_status(left, right, res_reuse1, ws);
        if (!s1.ok()) {
            std::cerr << "compute_status run 1 failed: " << s1.message << "\n";
            return 1;
        }
        const size_t retained = ws.allocated_bytes();
        if (retained == 0) {
            std::cerr << "Expected BoundedReuse to retain workspace buffers, got 0\n";
            return 1;
        }

        // Run second frame using the same workspace
        Status s2 = matcher.compute_status(left, right, res_reuse2, ws);
        if (!s2.ok()) {
            std::cerr << "compute_status run 2 failed: " << s2.message << "\n";
            return 1;
        }

        // Verify bit-exact consistency between repeated runs
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (res_reuse1.disparity.at(x, y) != res_reuse2.disparity.at(x, y)) {
                    std::cerr << "Disparity mismatch between sequential reuse runs at (" << x << "," << y << ")\n";
                    return 1;
                }
            }
        }

        // Test manual trim and release
        ws.trim(retained / 2);
        if (ws.allocated_bytes() != 0) {
            std::cerr << "Expected trim to release buffers when exceeding budget\n";
            return 1;
        }

        std::cout << "  BoundedReuse buffer retention and bit-exact reproducibility: PASS\n";
    }

    // 4. Test ResourceLimitExceeded enforcement
    {
        ExecutionOptions opt;
        opt.memory_policy = MemoryPolicy::BoundedReuse;
        opt.max_retained_bytes = 1024; // 1 KB budget, far smaller than required
        StereoWorkspace ws(opt);
        StereoResult res;

        Status s = matcher.compute_status(left, right, res, ws);
        if (s.code != StatusCode::ResourceLimitExceeded) {
            std::cerr << "Expected ResourceLimitExceeded, got: " << status_code_string(s.code) << "\n";
            return 1;
        }
        if (!res.empty()) {
            std::cerr << "Expected StereoResult to be cleared on ResourceLimitExceeded\n";
            return 1;
        }
        std::cout << "  ResourceLimitExceeded enforcement: PASS\n";
    }

    // 5. Test Bit-exact parity between default one-shot compute and workspace compute
    {
        StereoResult res_default;
        Status s_def = matcher.compute_status(left, right, res_default);
        if (!s_def.ok()) {
            std::cerr << "default compute_status failed\n";
            return 1;
        }

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (res_default.disparity.at(x, y) != res_reuse1.disparity.at(x, y)) {
                    std::cerr << "Default vs Workspace disparity mismatch at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (res_default.confidence.at(x, y) != res_reuse1.confidence.at(x, y)) {
                    std::cerr << "Default vs Workspace confidence mismatch at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (res_default.invalid_reason.at(x, y) != res_reuse1.invalid_reason.at(x, y)) {
                    std::cerr << "Default vs Workspace invalid_reason mismatch at (" << x << "," << y << ")\n";
                    return 1;
                }
                if (res_default.reliable_mask.at(x, y) != res_reuse1.reliable_mask.at(x, y)) {
                    std::cerr << "Default vs Workspace reliable_mask mismatch at (" << x << "," << y << ")\n";
                    return 1;
                }
            }
        }
        std::cout << "  Default one-shot vs Workspace 100% bit-exact parity: PASS\n";
    }

    std::cout << "All Workspace & Resource Policy tests passed!\n";
    return 0;
}
