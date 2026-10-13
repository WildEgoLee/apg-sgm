#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>

namespace apg {

struct StageTiming {
    double aux_ms = 0.0;
    double prior_ms = 0.0;
    double cost_ms = 0.0;
    double cross_ms = 0.0;
    double right_wta_ms = 0.0;
    double sgm_ms = 0.0;
    double wta_ms = 0.0;
    double confidence_ms = 0.0;
    double refine_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;
};

struct PipelineStats {
    StageTiming timing;

    size_t cost_bytes = 0;
    size_t aggregated_cost_bytes = 0;
    size_t estimated_peak_bytes = 0;
    size_t peak_workspace_bytes = 0;
    size_t retained_workspace_bytes = 0;
    size_t allocation_count = 0;
    size_t reuse_count = 0;

    int requested_threads = 0;
    int effective_threads = 0;
    std::string requested_backend = "Auto";
    std::string resolved_backend = "Dense";
    std::string memory_policy = "ReleasePerFrame";

    size_t prior_support_count = 0;
    double mean_search_width = 0.0;
    double mean_geometry_width = 0.0;
    double geometry_reduction_ratio = 0.0;
    double prior_incremental_reduction_ratio = 0.0;
    double search_reduction_ratio = 0.0;

    size_t valid_pixels = 0;
    size_t reliable_pixels = 0;
    size_t unreliable_pixels = 0;
    size_t refine_changed_pixels = 0;
    size_t lr_fail_pixels = 0;

    float valid_disparity_ratio = 0.0f;
    float reliable_ratio = 0.0f;
    float unreliable_ratio = 0.0f;
    float refine_changed_ratio = 0.0f;
    float lr_fail_ratio = 0.0f;

    std::string to_string() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "=== Pipeline Diagnostics & Stats ===\n";
        oss << "Timing (ms):\n";
        oss << "  Total:      " << timing.total_ms << "\n";
        oss << "  Aux:        " << timing.aux_ms << "\n";
        oss << "  Prior:      " << timing.prior_ms << "\n";
        oss << "  Cost:       " << timing.cost_ms << "\n";
        oss << "  Cross:      " << timing.cross_ms << "\n";
        oss << "  Right WTA:  " << timing.right_wta_ms << "\n";
        oss << "  SGM:        " << timing.sgm_ms << "\n";
        oss << "  WTA:        " << timing.wta_ms << "\n";
        oss << "  Confidence: " << timing.confidence_ms << "\n";
        oss << "  Refine:     " << timing.refine_ms << "\n";
        oss << "  Post:       " << timing.post_ms << "\n";
        oss << "Execution & Environment:\n";
        oss << "  Requested Threads:  " << requested_threads << "\n";
        oss << "  Effective Threads:  " << effective_threads << "\n";
        oss << "  Requested Backend:  " << requested_backend << "\n";
        oss << "  Resolved Backend:   " << resolved_backend << "\n";
        oss << "  Memory Policy:      " << memory_policy << "\n";
        oss << "  Allocations:        " << allocation_count << "\n";
        oss << "  Reuses:             " << reuse_count << "\n";
        oss << "Memory Accounting:\n";
        oss << "  Peak Workspace:     " << peak_workspace_bytes / (1024.0 * 1024.0) << " MB\n";
        oss << "  Retained Workspace: " << retained_workspace_bytes / (1024.0 * 1024.0) << " MB\n";
        oss << "Reliability & Quality:\n";
        oss << "  Valid Ratio:        " << (valid_disparity_ratio * 100.0f) << " %\n";
        oss << "  Reliable Ratio:     " << (reliable_ratio * 100.0f) << " %\n";
        oss << "  Unreliable Ratio:   " << (unreliable_ratio * 100.0f) << " %\n";
        oss << "  Refine Changed:     " << (refine_changed_ratio * 100.0f) << " %\n";
        oss << "  LR Fail Ratio:      " << (lr_fail_ratio * 100.0f) << " %\n";
        return oss.str();
    }

    std::string to_json() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "{\n";
        oss << "  \"execution\": {\n";
        oss << "    \"requested_threads\": " << requested_threads << ",\n";
        oss << "    \"effective_threads\": " << effective_threads << ",\n";
        oss << "    \"requested_backend\": \"" << requested_backend << "\",\n";
        oss << "    \"resolved_backend\": \"" << resolved_backend << "\",\n";
        oss << "    \"memory_policy\": \"" << memory_policy << "\",\n";
        oss << "    \"allocation_count\": " << allocation_count << ",\n";
        oss << "    \"reuse_count\": " << reuse_count << "\n";
        oss << "  },\n";
        oss << "  \"timing\": {\n";
        oss << "    \"total_ms\": " << timing.total_ms << ",\n";
        oss << "    \"aux_ms\": " << timing.aux_ms << ",\n";
        oss << "    \"prior_ms\": " << timing.prior_ms << ",\n";
        oss << "    \"cost_ms\": " << timing.cost_ms << ",\n";
        oss << "    \"cross_ms\": " << timing.cross_ms << ",\n";
        oss << "    \"right_wta_ms\": " << timing.right_wta_ms << ",\n";
        oss << "    \"sgm_ms\": " << timing.sgm_ms << ",\n";
        oss << "    \"wta_ms\": " << timing.wta_ms << ",\n";
        oss << "    \"confidence_ms\": " << timing.confidence_ms << ",\n";
        oss << "    \"refine_ms\": " << timing.refine_ms << ",\n";
        oss << "    \"post_ms\": " << timing.post_ms << "\n";
        oss << "  },\n";
        oss << "  \"memory\": {\n";
        oss << "    \"peak_workspace_bytes\": " << peak_workspace_bytes << ",\n";
        oss << "    \"retained_workspace_bytes\": " << retained_workspace_bytes << "\n";
        oss << "  },\n";
        oss << "  \"quality\": {\n";
        oss << "    \"valid_disparity_ratio\": " << valid_disparity_ratio << ",\n";
        oss << "    \"reliable_ratio\": " << reliable_ratio << ",\n";
        oss << "    \"unreliable_ratio\": " << unreliable_ratio << ",\n";
        oss << "    \"refine_changed_ratio\": " << refine_changed_ratio << ",\n";
        oss << "    \"lr_fail_ratio\": " << lr_fail_ratio << "\n";
        oss << "  }\n";
        oss << "}";
        return oss.str();
    }
};

} // namespace apg
