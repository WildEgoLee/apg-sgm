#include "apg_sgm/ablation.hpp"

#include <algorithm>
#include <cctype>

namespace apg {

PipelineConfig make_ablation_config(AblationId id, int max_disparity) {
    PipelineConfig c;
    c.min_disparity = 0;
    c.max_disparity = max_disparity;
    c.post.subpixel = true;
    c.post.lr_check = true;
    c.post.lr_max_diff = 1;
    c.post.median_radius = 1;
    c.post.max_fill_gap = 32;

    switch (id) {
    case AblationId::A_Census4:
        c.cost.use_ad = false;
        c.cost.use_grad = false;
        c.aggregation.enable = false;
        c.sgm.adaptive_penalties = false;
        c.sgm.paths = PathType::Path4;
        c.prior.enable = false;
        c.refine.enable = false;
        break;

    case AblationId::B_MultiCost4:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = false;
        c.sgm.adaptive_penalties = false;
        c.sgm.paths = PathType::Path4;
        c.prior.enable = false;
        c.refine.enable = false;
        break;

    case AblationId::C_Cross4:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.aggregation.iterations = 1;
        c.sgm.adaptive_penalties = false;
        c.sgm.paths = PathType::Path4;
        c.prior.enable = false;
        c.refine.enable = false;
        break;

    case AblationId::D_Adaptive4:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.aggregation.iterations = 1;
        c.sgm.adaptive_penalties = true;
        c.sgm.paths = PathType::Path4;
        c.prior.enable = false;
        c.refine.enable = false;
        break;

    case AblationId::E_Prior4:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.aggregation.iterations = 1;
        c.sgm.adaptive_penalties = true;
        c.sgm.paths = PathType::Path4;
        c.prior.enable = true;
        c.refine.enable = false;
        break;

    case AblationId::F_Refine4:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.aggregation.iterations = 1;
        c.sgm.adaptive_penalties = true;
        c.sgm.paths = PathType::Path4;
        c.prior.enable = true;
        c.refine.enable = true;
        c.refine.iterations = 3;
        break;

    case AblationId::G_HQ8:
        c.cost.use_ad = true;
        c.cost.use_grad = true;
        c.aggregation.enable = true;
        c.aggregation.iterations = 1;
        c.sgm.adaptive_penalties = true;
        c.sgm.paths = PathType::Path8;
        c.prior.enable = true;
        c.refine.enable = true;
        c.refine.iterations = 3;
        break;
    }

    return c;
}

std::string ablation_id_to_string(AblationId id) {
    switch (id) {
    case AblationId::A_Census4: return "A";
    case AblationId::B_MultiCost4: return "B";
    case AblationId::C_Cross4: return "C";
    case AblationId::D_Adaptive4: return "D";
    case AblationId::E_Prior4: return "E";
    case AblationId::F_Refine4: return "F";
    case AblationId::G_HQ8: return "G";
    }
    return "A";
}

std::string ablation_id_description(AblationId id) {
    switch (id) {
    case AblationId::A_Census4: return "Census + 4SGM";
    case AblationId::B_MultiCost4: return "Census+AD+Grad + 4SGM";
    case AblationId::C_Cross4: return "Census+AD+Grad + Cross + 4SGM";
    case AblationId::D_Adaptive4: return "Census+AD+Grad + Cross + AdaptiveP2 + 4SGM";
    case AblationId::E_Prior4: return "AdaptiveP2 + Prior + 4SGM";
    case AblationId::F_Refine4: return "Prior + Refiner + 4SGM";
    case AblationId::G_HQ8: return "Prior + Refiner + 8SGM (HQ)";
    }
    return "";
}

bool parse_ablation_id(const std::string& str, AblationId& out) {
    std::string s = str;
    for (char& ch : s) ch = static_cast<char>(std::toupper(ch));

    if (s == "A" || s == "A_CENSUS4") { out = AblationId::A_Census4; return true; }
    if (s == "B" || s == "B_MULTICOST4") { out = AblationId::B_MultiCost4; return true; }
    if (s == "C" || s == "C_CROSS4") { out = AblationId::C_Cross4; return true; }
    if (s == "D" || s == "D_ADAPTIVE4") { out = AblationId::D_Adaptive4; return true; }
    if (s == "E" || s == "E_PRIOR4") { out = AblationId::E_Prior4; return true; }
    if (s == "F" || s == "F_REFINE4") { out = AblationId::F_Refine4; return true; }
    if (s == "G" || s == "G_HQ8") { out = AblationId::G_HQ8; return true; }
    return false;
}

std::vector<AblationId> all_ablation_ids() {
    return {
        AblationId::A_Census4,
        AblationId::B_MultiCost4,
        AblationId::C_Cross4,
        AblationId::D_Adaptive4,
        AblationId::E_Prior4,
        AblationId::F_Refine4,
        AblationId::G_HQ8
    };
}

} // namespace apg
