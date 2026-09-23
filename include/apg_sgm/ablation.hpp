#pragma once

#include "apg_sgm/config.hpp"
#include <string>
#include <vector>

namespace apg {

enum class AblationId {
    A_Census4,
    B_MultiCost4,
    C_Cross4,
    D_Adaptive4,
    E_Prior4,
    F_Refine4,
    G_HQ8
};

PipelineConfig make_ablation_config(AblationId id, int max_disparity = 128);

std::string ablation_id_to_string(AblationId id);
std::string ablation_id_description(AblationId id);
bool parse_ablation_id(const std::string& str, AblationId& out);
std::vector<AblationId> all_ablation_ids();

} // namespace apg
