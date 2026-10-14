#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/cost_computer.hpp"
#include "hierarchical_stereo.hpp"

#include <cassert>
#include <iostream>

using namespace apg;

static void test_scope_isolation() {
    assert(!ExperimentalHierarchicalScope::is_enabled());

    {
        HierarchicalConfig cfg;
        cfg.enable = true;
        cfg.radius = 8;
        cfg.fallback = HierarchicalFallback::NearestValid;
        ExperimentalHierarchicalScope scope(cfg);

        assert(ExperimentalHierarchicalScope::is_enabled());
        assert(ExperimentalHierarchicalScope::current_config().radius == 8);
        assert(ExperimentalHierarchicalScope::current_config().fallback == HierarchicalFallback::NearestValid);
    }

    assert(!ExperimentalHierarchicalScope::is_enabled());
    std::cout << "[PASS] test_scope_isolation\n";
}

static void test_hierarchical_search_range() {
    const int w = 128;
    const int h = 96;

    Image8 left(w, h, 1);
    Image8 right(w, h, 1);

    // Textured synthetic pattern with a known disparity shift of 8 pixels
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t pattern = static_cast<uint8_t>((x * 37 + y * 53 + (x ^ y) * 17) % 256);
            left.at(x, y) = pattern;
            int xr = x - 8;
            if (xr >= 0 && xr < w) {
                right.at(xr, y) = pattern;
            } else {
                right.at(x, y) = pattern;
            }
        }
    }

    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced, 32);
    cfg.min_disparity = 0;
    cfg.max_disparity = 32;

    PipelineBuffers buf;
    CostComputer cost;
    cost.compute_aux(left, right, cfg, buf);

    HierarchicalConfig hcfg;
    hcfg.enable = true;
    hcfg.radius = 4;
    hcfg.fallback = HierarchicalFallback::NearestValid;

    HierarchicalTelemetry telem;
    bool ok = HierarchicalStereo::estimate_search_range(hcfg, cfg, buf, &telem);
    assert(ok);

    assert(telem.avg_states_per_pixel > 0.0);
    assert(telem.avg_states_per_pixel < 32.0); // Substantial reduction from 32
    assert(telem.coarse_valid_ratio >= 0.0f && telem.coarse_valid_ratio <= 1.0f);

    // Verify all ranges are valid
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int lo = buf.range.dmin.at(x, y);
            int hi = buf.range.dmax.at(x, y);
            assert(lo >= cfg.min_disparity);
            assert(hi <= cfg.max_disparity);
            assert(lo <= hi);
        }
    }

    std::cout << "[PASS] test_hierarchical_search_range (avg_states=" << telem.avg_states_per_pixel << ")\n";
}

static void test_hierarchical_end_to_end() {
    const int w = 128;
    const int h = 96;

    Image8 left(w, h, 1);
    Image8 right(w, h, 1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t v = static_cast<uint8_t>((x * 19 + y * 23) % 256);
            left.at(x, y) = v;
            right.at(x, y) = v;
        }
    }

    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced, 32);
    cfg.volume_backend = VolumeBackend::Packed;

    StereoMatcher matcher(cfg);
    StereoResult result;
    StereoWorkspace ws;
    PipelineStats stats;

    HierarchicalConfig hcfg;
    hcfg.enable = true;
    hcfg.radius = 4;
    hcfg.fallback = HierarchicalFallback::NearestValid;
    ExperimentalHierarchicalScope scope(hcfg);

    Status s = matcher.compute_status(left, right, result, ws, &stats);
    assert(s.ok());
    assert(!result.disparity.empty());
    assert(stats.peak_workspace_bytes > 0);

    const auto& telem = ExperimentalHierarchicalScope::current_telemetry();
    assert(telem.avg_states_per_pixel > 0.0);
    assert(telem.avg_states_per_pixel < 32.0);

    std::cout << "[PASS] test_hierarchical_end_to_end\n";
}

int main() {
    test_scope_isolation();
    test_hierarchical_search_range();
    test_hierarchical_end_to_end();
    std::cout << "All hierarchical stereo tests passed successfully.\n";
    return 0;
}
