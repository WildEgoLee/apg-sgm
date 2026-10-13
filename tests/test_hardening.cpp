#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/session.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using namespace apg;

static void test_invalid_disparity_range() {
    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced);
    cfg.min_disparity = 64;
    cfg.max_disparity = 32; // Invalid: max <= min
    StereoMatcher matcher(cfg);

    Image8 left(64, 48, 1);
    Image8 right(64, 48, 1);
    StereoResult result;
    StereoWorkspace ws;

    Status s = matcher.compute_status(left, right, result, ws);
    assert(!s.ok());
    assert(s.code == StatusCode::InvalidDisparityRange);
    std::cout << "[PASS] test_invalid_disparity_range\n";
}

static void test_invalid_stride() {
    StereoSession session;
    uint8_t dummy[100];
    // Invalid stride: width=32, channels=1, but stride=16 (< width * channels)
    ImageView8 left(dummy, 32, 24, 16, 1);
    ImageView8 right(dummy, 32, 24, 16, 1);
    StereoResult result;

    Status s = session.process_status(left, right, result);
    assert(!s.ok());
    assert(s.code == StatusCode::SizeMismatch);
    std::cout << "[PASS] test_invalid_stride\n";
}

static void test_precompute_budget_gate() {
    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Fast, 64);
    StereoMatcher matcher(cfg);

    Image8 left(320, 240, 1);
    Image8 right(320, 240, 1);
    left.fill(128);
    right.fill(128);

    // Set budget to 10 KB, while 320x240 with dmax=64 requires several megabytes
    ExecutionOptions opts;
    opts.max_retained_bytes = 10 * 1024;
    StereoWorkspace ws(opts);
    StereoResult result;

    Status s = matcher.compute_status(left, right, result, ws);
    assert(!s.ok());
    assert(s.code == StatusCode::ResourceLimitExceeded);
    // Crucial check: workspace allocated bytes must be 0 because compute aborted BEFORE allocating!
    assert(ws.allocated_bytes() == 0);
    std::cout << "[PASS] test_precompute_budget_gate\n";
}

static void test_session_fault_recovery() {
    StereoSession session;
    Image8 left(64, 48, 1);
    Image8 right(64, 48, 1);
    left.fill(100);
    right.fill(100);

    StereoResult result;

    // Frame 1: Valid input
    Status s1 = session.process_status(left, right, result);
    assert(s1.ok());
    assert(session.frame_count() == 1);

    // Frame 2: Invalid input (empty)
    Image8 empty_img;
    Status s2 = session.process_status(empty_img, right, result);
    assert(!s2.ok());
    assert(s2.code == StatusCode::EmptyInput);
    assert(session.frame_count() == 1); // Frame count did not advance on failure

    // Frame 3: Valid input again -> Session recovers cleanly
    Status s3 = session.process_status(left, right, result);
    assert(s3.ok());
    assert(session.frame_count() == 2);
    assert(!result.empty());
    std::cout << "[PASS] test_session_fault_recovery\n";
}

static void test_bounded_reuse_steady_state() {
    ExecutionOptions opts;
    opts.memory_policy = MemoryPolicy::BoundedReuse;
    StereoSession session(PipelineConfig::from_mode(QualityMode::Fast, 32), opts);

    Image8 left(64, 48, 1);
    Image8 right(64, 48, 1);
    left.fill(150);
    right.fill(150);

    StereoResult result;
    PipelineStats stats1;
    Status s1 = session.process_status(left, right, result, &stats1);
    assert(s1.ok());
    assert(stats1.allocation_count == 1);
    assert(stats1.reuse_count == 0);
    assert(stats1.frame_allocation_count == 1);
    assert(stats1.frame_reuse_count == 0);

    // Frame 2: Same dimension and configuration -> Must reuse existing workspace allocations!
    PipelineStats stats2;
    Status s2 = session.process_status(left, right, result, &stats2);
    assert(s2.ok());
    assert(stats2.allocation_count == 1);
    assert(stats2.reuse_count == 1);
    assert(stats2.frame_allocation_count == 1);
    assert(stats2.frame_reuse_count == 1);

    // Frame 3: Still same dimensions -> reuse increments again
    PipelineStats stats3;
    Status s3 = session.process_status(left, right, result, &stats3);
    assert(s3.ok());
    assert(stats3.allocation_count == 1);
    assert(stats3.reuse_count == 2);
    assert(stats3.frame_allocation_count == 1);
    assert(stats3.frame_reuse_count == 2);

    std::cout << "[PASS] test_bounded_reuse_steady_state\n";
}

static void test_retained_memory_large_to_small_accounting() {
    ExecutionOptions opts;
    opts.memory_policy = MemoryPolicy::BoundedReuse;
    StereoSession session(PipelineConfig::from_mode(QualityMode::Fast, 32), opts);

    Image8 left_large(128, 96, 1);
    Image8 right_large(128, 96, 1);
    left_large.fill(120);
    right_large.fill(120);

    StereoResult result;
    PipelineStats stats_large;
    Status s1 = session.process_status(left_large, right_large, result, &stats_large);
    assert(s1.ok());
    const size_t large_bytes = session.allocated_bytes();
    assert(large_bytes > 0);
    assert(stats_large.retained_workspace_bytes == large_bytes);

    // Now process a smaller frame: capacities should be retained, not truncated
    Image8 left_small(64, 48, 1);
    Image8 right_small(64, 48, 1);
    left_small.fill(120);
    right_small.fill(120);

    PipelineStats stats_small;
    Status s2 = session.process_status(left_small, right_small, result, &stats_small);
    assert(s2.ok());
    const size_t small_bytes = session.allocated_bytes();
    assert(stats_small.retained_workspace_bytes == small_bytes);

    // Critical assertion: retained memory accounting must reflect actual capacity,
    // so small_bytes must be >= large_bytes (no under-reporting of retained heap capacity).
    assert(small_bytes >= large_bytes);
    std::cout << "[PASS] test_retained_memory_large_to_small_accounting\n";
}

static void test_estimate_workspace_overflow() {
    PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Fast, 64);
    // Extreme dimensions that overflow uint64_t multiplication
    size_t est = estimate_workspace_bytes(10000000, 10000000, cfg);
    assert(est == SIZE_MAX);

    // Negative / zero inputs
    assert(estimate_workspace_bytes(-1, 100, cfg) == 0);
    assert(estimate_workspace_bytes(100, 0, cfg) == 0);
    std::cout << "[PASS] test_estimate_workspace_overflow\n";
}

int main() {
    test_invalid_disparity_range();
    test_invalid_stride();
    test_precompute_budget_gate();
    test_session_fault_recovery();
    test_bounded_reuse_steady_state();
    test_retained_memory_large_to_small_accounting();
    test_estimate_workspace_overflow();
    std::cout << "All hardening tests passed successfully!\n";
    return 0;
}
