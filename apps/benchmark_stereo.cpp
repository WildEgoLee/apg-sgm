#include "apg_sgm/ablation.hpp"
#include "apg_sgm/image.hpp"
#include "apg_sgm/metrics.hpp"
#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/pipeline_stats.hpp"
#include "hierarchical_stereo.hpp"
#include "subpixel_estimator.hpp"
#include "inpaint_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace fs = std::filesystem;

struct BenchmarkCase {
    std::string name;
    std::string left_path;
    std::string right_path;
    std::string gt_path;
    int dmax = 128;
    int dmin = 0;
    std::string vis_path;
};

template <typename ImgT>
static bool compare_images(const std::string& buffer_name, const ImgT& img1, const ImgT& img2) {
    if (img1.width() != img2.width() || img1.height() != img2.height()) {
        std::cerr << "Verification FAILED: " << buffer_name << " dimensions differ ("
                  << img1.width() << "x" << img1.height() << " vs "
                  << img2.width() << "x" << img2.height() << ")\n";
        return false;
    }
    if (img1.empty() && img2.empty()) return true;
    int mismatches = 0;
    for (int y = 0; y < img1.height(); ++y) {
        for (int x = 0; x < img1.width(); ++x) {
            auto v1 = img1.at(x, y);
            auto v2 = img2.at(x, y);
            bool diff = false;
            using ValT = decltype(v1);
            if constexpr (std::is_floating_point_v<ValT>) {
                uint32_t b1 = 0, b2 = 0;
                static_assert(sizeof(b1) == sizeof(v1));
                std::memcpy(&b1, &v1, sizeof(b1));
                std::memcpy(&b2, &v2, sizeof(b2));
                diff = (b1 != b2);
            } else {
                diff = (v1 != v2);
            }
            if (diff) {
                if (mismatches < 5) {
                    std::cerr << "Mismatch in " << buffer_name << " at (" << x << ", " << y
                              << "): dense=" << +v1 << ", packed=" << +v2 << "\n";
                }
                ++mismatches;
            }
        }
    }
    if (mismatches > 0) {
        std::cerr << "Total mismatches in " << buffer_name << ": " << mismatches
                  << " / " << (img1.width() * img1.height()) << "\n";
        return false;
    }
    return true;
}

static std::vector<BenchmarkCase> parse_manifest(const std::string& manifest_path) {
    std::vector<BenchmarkCase> cases;
    std::ifstream ifs(manifest_path);
    if (!ifs) {
        std::cerr << "Error: cannot open manifest file: " << manifest_path << std::endl;
        return cases;
    }

    fs::path base_dir = fs::path(manifest_path).parent_path();

    std::string line;
    while (std::getline(ifs, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') {
            continue;
        }

        std::istringstream iss(line.substr(start));
        BenchmarkCase c;
        if (!(iss >> c.name >> c.left_path >> c.right_path >> c.gt_path)) {
            continue;
        }

        iss >> c.dmax;
        iss >> c.dmin;
        iss >> c.vis_path;

        fs::path lp(c.left_path);
        if (lp.is_relative()) c.left_path = (base_dir / lp).lexically_normal().string();

        fs::path rp(c.right_path);
        if (rp.is_relative()) c.right_path = (base_dir / rp).lexically_normal().string();

        fs::path gp(c.gt_path);
        if (gp.is_relative()) c.gt_path = (base_dir / gp).lexically_normal().string();

        if (!c.vis_path.empty()) {
            fs::path vp(c.vis_path);
            if (vp.is_relative()) c.vis_path = (base_dir / vp).lexically_normal().string();
        }

        cases.push_back(c);
    }
    return cases;
}

int main(int argc, char** argv) {
    std::string manifest_path;
    std::string ablation_arg = "all";
    std::string output_path = "results/ablation.csv";
    std::string save_disp_dir;
    int warmup_runs = 2;
    int repeat_runs = 5;
    int num_threads = 0;
    std::string backend_arg = "auto";
    bool verify_backends = false;
    bool hierarchical_enable = false;
    int hierarchical_radius = 8;
    apg::HierarchicalFallback hierarchical_fallback = apg::HierarchicalFallback::NearestValid;
    bool hierarchical_coarse_cross = true;
    bool hierarchical_coarse_fill = true;
    bool hierarchical_adaptive = false;
    float hierarchical_confidence_threshold = 0.5f;
    int hierarchical_expanded_radius = 8;
    apg::SubpixelMethod subpixel_method = apg::SubpixelMethod::Legacy;
    float subpixel_param = 0.0f;
    bool subpixel_param_set = false;
    bool subpixel_telem = false;
    apg::InpaintMethod inpaint_method = apg::InpaintMethod::Legacy;
    int inpaint_gap = 32;
    bool inpaint_telem = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if ((arg == "--ablation" || arg == "--modes") && i + 1 < argc) {
            ablation_arg = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--save_disp" && i + 1 < argc) {
            save_disp_dir = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat_runs = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--backend" && i + 1 < argc) {
            backend_arg = argv[++i];
        } else if (arg == "--packed") {
            backend_arg = "packed";
        } else if (arg == "--verify-backends") {
            verify_backends = true;
        } else if (arg == "--hierarchical") {
            hierarchical_enable = true;
        } else if (arg == "--hierarchical-radius" && i + 1 < argc) {
            hierarchical_radius = std::stoi(argv[++i]);
        } else if (arg == "--hierarchical-fallback" && i + 1 < argc) {
            std::string fstr = argv[++i];
            if (!apg::parse_hierarchical_fallback(fstr, hierarchical_fallback)) {
                std::cerr << "Error: invalid hierarchical fallback policy '" << fstr
                          << "'. Options: full_range, nearest_valid, expanded_radius\n";
                return 1;
            }
        } else if (arg == "--hierarchical-no-cross") {
            hierarchical_coarse_cross = false;
        } else if (arg == "--hierarchical-no-fill") {
            hierarchical_coarse_fill = false;
        } else if (arg == "--hierarchical-adaptive") {
            hierarchical_adaptive = true;
        } else if (arg == "--hierarchical-confidence-threshold" && i + 1 < argc) {
            hierarchical_confidence_threshold = std::stof(argv[++i]);
        } else if (arg == "--hierarchical-expanded-radius" && i + 1 < argc) {
            hierarchical_expanded_radius = std::stoi(argv[++i]);
        } else if (arg == "--subpixel-method" && i + 1 < argc) {
            std::string mstr = argv[++i];
            if (!apg::parse_subpixel_method(mstr, subpixel_method)) {
                std::cerr << "Error: invalid subpixel method '" << mstr
                          << "'. Options: legacy, curv_gate, conf_gate, stabilized\n";
                return 1;
            }
        } else if (arg == "--subpixel-param" && i + 1 < argc) {
            subpixel_param = std::stof(argv[++i]);
            subpixel_param_set = true;
        } else if (arg == "--subpixel-telem") {
            subpixel_telem = true;
        } else if (arg == "--inpaint-method" && i + 1 < argc) {
            std::string mstr = argv[++i];
            if (!apg::parse_inpaint_method(mstr, inpaint_method)) {
                std::cerr << "Error: invalid inpaint method '" << mstr
                          << "'. Options: legacy, shadow_reclass\n";
                return 1;
            }
        } else if (arg == "--inpaint-gap" && i + 1 < argc) {
            inpaint_gap = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--inpaint-telem") {
            inpaint_telem = true;
        } else if (arg == "--preset" && i + 1 < argc) {
            std::string pstr = argv[++i];
            if (pstr == "quality_v2" || pstr == "v2" || pstr == "QualityV2") {
                ablation_arg = "F";
                hierarchical_enable = true;
                hierarchical_radius = 4;
                hierarchical_fallback = apg::HierarchicalFallback::NearestValid;
                inpaint_method = apg::InpaintMethod::ShadowReclass;
                inpaint_gap = 48;
            } else {
                std::cerr << "Error: invalid preset '" << pstr << "'. Options: quality_v2\n";
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: apg-benchmark [options]\n"
                      << "Options:\n"
                      << "  --manifest <path>     Path to dataset manifest file (required)\n"
                      << "  --ablation, --modes <id|all>\n"
                      << "                        Ablation experiment: A, B, C, D, E, F, G, or all (default: all)\n"
                      << "  --preset <quality_v2> Preset configuration (e.g. quality_v2: Mode F + Hierarchical R=4 + Shadow Reclass gap=48)\n"
                      << "  --output <path>       Output CSV file path (default: results/ablation.csv)\n"
                      << "  --save_disp <dir>     Optional directory to save output disparity maps\n"
                      << "  --warmup <N>          Warmup runs before timing (default: 2)\n"
                      << "  --repeat <N>          Timed runs per test case (default: 5, reports median)\n"
                      << "  --threads <N>         Number of OpenMP threads (default: 0 = auto)\n"
                      << "  --backend <auto|dense|packed>\n"
                      << "                        Volume backend policy (default: auto -> packed for Prior/HQ, dense otherwise)\n"
                      << "  --packed              Alias for --backend packed\n"
                      << "  --verify-backends     Verify 100% bit-exact parity between dense and packed backends\n"
                      << "  --hierarchical        Enable hierarchical coarse-to-fine search space reduction\n"
                      << "  --hierarchical-radius <N>\n"
                      << "                        Radius R around coarse disparity (default: 8, options: 4, 8, 12, 16)\n"
                      << "  --hierarchical-fallback <policy>\n"
                      << "                        Fallback policy (full_range, nearest_valid, expanded_radius)\n"
                      << "  --hierarchical-no-cross\n"
                      << "                        Disable cross-aggregation at coarse level\n"
                      << "  --hierarchical-no-fill\n"
                      << "                        Disable hole filling at coarse level\n"
                      << "  --hierarchical-adaptive\n"
                      << "                        Enable confidence-aware adaptive search radius\n"
                      << "  --hierarchical-confidence-threshold <float>\n"
                      << "                        Confidence threshold for adaptive radius (default: 0.5)\n"
                      << "  --hierarchical-expanded-radius <N>\n"
                      << "                        Expanded radius for low-confidence pixels (default: 8)\n"
                      << "  --subpixel-method <legacy|curv_gate|conf_gate|stabilized>\n"
                      << "                        Subpixel refinement method (default: legacy)\n"
                      << "  --subpixel-param <float>\n"
                      << "                        Method-specific parameter (curvature threshold, confidence threshold, or lambda)\n"
                      << "  --subpixel-telem      Collect and display subpixel diagnostics telemetry\n"
                      << "  --inpaint-method <legacy|shadow_reclass>\n"
                      << "                        Inpainting / hole filling method (default: legacy)\n"
                      << "  --inpaint-gap <N>     Maximum gap distance for hole search (default: 32)\n"
                      << "  --inpaint-telem       Collect and display inpainting attribution telemetry\n"
                      << "  --help, -h            Show this help message\n";
            return 0;
        } else {
            std::cerr << "Error: unknown or incomplete argument '" << arg << "'. Use --help for usage.\n";
            return 1;
        }
    }

    if (!subpixel_param_set) {
        if (subpixel_method == apg::SubpixelMethod::CurvGate) {
            subpixel_param = 4.0f;
        } else if (subpixel_method == apg::SubpixelMethod::ConfGate) {
            subpixel_param = 0.05f;
        } else if (subpixel_method == apg::SubpixelMethod::Stabilized) {
            subpixel_param = 4.0f;
        }
    }

    apg::VolumeBackend volume_backend = apg::VolumeBackend::Auto;
    if (backend_arg == "packed" || backend_arg == "PACKED") {
        volume_backend = apg::VolumeBackend::Packed;
    } else if (backend_arg == "dense" || backend_arg == "DENSE") {
        volume_backend = apg::VolumeBackend::Dense;
    } else if (backend_arg == "auto" || backend_arg == "AUTO") {
        volume_backend = apg::VolumeBackend::Auto;
    } else {
        std::cerr << "Error: unrecognized --backend '" << backend_arg << "'. Options: auto, dense, packed.\n";
        return 1;
    }

    if (manifest_path.empty()) {
        std::cerr << "Error: --manifest is required. Run with --help for details.\n";
        return 1;
    }

    auto cases = parse_manifest(manifest_path);
    if (cases.empty()) {
        std::cerr << "Error: no valid test cases found in manifest: " << manifest_path << "\n";
        return 1;
    }

    std::vector<apg::AblationId> ablation_ids;
    if (ablation_arg == "all" || ablation_arg == "ALL") {
        ablation_ids = apg::all_ablation_ids();
    } else {
        std::stringstream ss(ablation_arg);
        std::string token;
        while (std::getline(ss, token, ',')) {
            while (!token.empty() && std::isspace(token.front())) token.erase(token.begin());
            while (!token.empty() && std::isspace(token.back())) token.pop_back();
            if (!token.empty()) {
                apg::AblationId id;
                if (apg::parse_ablation_id(token, id)) {
                    ablation_ids.push_back(id);
                } else {
                    std::cerr << "Warning: unrecognized ablation ID: " << token << "\n";
                }
            }
        }
    }

    if (ablation_ids.empty()) {
        std::cerr << "Error: no valid ablation configurations selected.\n";
        return 1;
    }

    if (verify_backends) {
        std::cout << "============================================================\n"
                  << "Verifying Dual Backend Parity (Dense vs Packed)\n"
                  << "Manifest: " << manifest_path << " (" << cases.size() << " cases)\n"
                  << "Configurations: " << ablation_ids.size() << " ablation modes\n"
                  << "Threads: " << num_threads << "\n"
                  << "============================================================\n";

        bool all_passed = true;
        for (const auto& c : cases) {
            apg::Image8 left_img, right_img;
            if (!apg::load_image(c.left_path, left_img) || !apg::load_image(c.right_path, right_img)) {
                std::cerr << "Error loading input images for case: " << c.name << "\n";
                return 1;
            }

            for (const auto& aid : ablation_ids) {
                std::string aid_str = apg::ablation_id_to_string(aid);
                std::cout << "Testing case " << c.name << " [" << aid_str << "] ... " << std::flush;

                apg::PipelineConfig cfg_dense = apg::make_ablation_config(aid, c.dmax);
                cfg_dense.min_disparity = c.dmin;
                cfg_dense.max_disparity = c.dmax;
                cfg_dense.volume_backend = apg::VolumeBackend::Dense;
                if (num_threads > 0) cfg_dense.num_threads = num_threads;

                if (hierarchical_enable) {
                    cfg_dense.hierarchical.enable = true;
                    cfg_dense.hierarchical.radius = hierarchical_radius;
                    cfg_dense.hierarchical.fallback = hierarchical_fallback;
                    cfg_dense.hierarchical.coarse_cross_agg = hierarchical_coarse_cross;
                    cfg_dense.hierarchical.coarse_fill_holes = hierarchical_coarse_fill;
                    cfg_dense.hierarchical.adaptive_radius = hierarchical_adaptive;
                    cfg_dense.hierarchical.confidence_threshold = hierarchical_confidence_threshold;
                    cfg_dense.hierarchical.expanded_radius = hierarchical_expanded_radius;
                }

                cfg_dense.post.max_fill_gap = inpaint_gap;

                apg::PipelineConfig cfg_packed = cfg_dense;
                cfg_packed.volume_backend = apg::VolumeBackend::Packed;

                apg::StereoMatcher matcher_dense(cfg_dense);
                apg::StereoMatcher matcher_packed(cfg_packed);

                apg::PipelineBuffers bufs_dense;
                apg::PipelineBuffers bufs_packed;

                apg::ExperimentalHierarchicalScope hscope(cfg_dense.hierarchical);
                apg::ExperimentalSubpixelScope sp_scope(subpixel_method, subpixel_param, false);
                apg::ExperimentalInpaintScope ip_scope(inpaint_method, inpaint_gap, false);

                if (!matcher_dense.compute(left_img, right_img, bufs_dense, nullptr)) {
                    std::cerr << "Dense failed on case " << c.name << ": "
                              << matcher_dense.last_error() << "\n";
                    return 1;
                }
                if (!matcher_packed.compute(left_img, right_img, bufs_packed, nullptr)) {
                    std::cerr << "Packed failed on case " << c.name << ": "
                              << matcher_packed.last_error() << "\n";
                    return 1;
                }

                bool pass = true;
                pass &= compare_images("disparity", bufs_dense.disparity, bufs_packed.disparity);
                pass &= compare_images("disparity_right", bufs_dense.disparity_right, bufs_packed.disparity_right);
                pass &= compare_images("d_before_refine", bufs_dense.d_before_refine, bufs_packed.d_before_refine);
                pass &= compare_images("d_after_refine", bufs_dense.d_after_refine, bufs_packed.d_after_refine);
                pass &= compare_images("confidence", bufs_dense.confidence, bufs_packed.confidence);
                pass &= compare_images("reliable_mask", bufs_dense.reliable_mask, bufs_packed.reliable_mask);
                pass &= compare_images("invalid_reason", bufs_dense.invalid_reason, bufs_packed.invalid_reason);

                if (!pass) {
                    std::cout << "FAILED!\n";
                    all_passed = false;
                } else {
                    std::cout << "PASS (100% bit-exact parity across all buffers)\n";
                }
            }
        }

        if (!all_passed) {
            std::cerr << "\nVerification FAILED on one or more cases!\n";
            return 1;
        }

        std::cout << "\nAll test cases and ablation modes passed bit-exact dual backend parity!\n";
        return 0;
    }

    // Ensure output directory exists
    fs::path out_file_path(output_path);
    if (out_file_path.has_parent_path()) {
        fs::create_directories(out_file_path.parent_path());
    }

    if (!save_disp_dir.empty()) {
        fs::create_directories(save_disp_dir);
    }

    std::ofstream csv(output_path);
    if (!csv) {
        std::cerr << "Error: cannot open output CSV file: " << output_path << "\n";
        return 1;
    }

#if defined(NDEBUG)
    const std::string build_type = "Release";
#else
    const std::string build_type = "Debug";
#endif

    int effective_threads = 1;
#if defined(_OPENMP)
    effective_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
#endif

    csv << "backend,threads,effective_threads,warmup,repeat,build_type,"
        << "case,ablation_id,ablation_name,cost,cross,p2,prior,refine,paths,w,h,dmax,"
        << "epe,bad_0_5,bad_1_0,bad_2_0,bad_3_0,kitti_d1_all,kitti_d1_noc,valid_ratio,lr_fail_ratio,edge_epe,nonedge_epe,"
        << "recall_all,recall_matchable,recall_visible,range_vis_eval_px,range_vis_in_px,range_mat_eval_px,range_mat_in_px,"
        << "prior_miss_vis_count,prior_miss_edge_pct,prior_miss_nonedge_pct,"
        << "prior_supports,support_p05,support_p1,support_p2,support_p1_vis,support_pvis_05,support_pvis_1,support_pvis_2,"
        << "support_gt_valid,support_visible,support_correct_05,support_correct_1,support_correct_2,"
        << "support_visible_correct_05,support_visible_correct_1,support_visible_correct_2,"
        << "support_visible_gt_cells,support_correct_visible_cells,"
        << "support_mae,support_mae_vis,grid_recall_1,"
        << "mean_search_width,mean_geom_width,geom_reduction_ratio,prior_incremental_reduction,total_reduction_ratio,"
        << "cost_bytes,cost32_bytes,peak_bytes,"
        << "reliable_ratio,unreliable_ratio,refine_changed_ratio,refine_epe_delta,post_epe_delta,total_epe_delta,"
        << "time_total_ms,time_aux_ms,time_prior_ms,time_cost_ms,time_right_wta_ms,time_cross_ms,time_sgm_ms,time_wta_ms,time_confidence_ms,time_refine_ms,time_post_ms,"
        << "all_gt_pixels,all_valid_est_pixels,all_valid_ratio,all_epe,all_bad05,all_bad10,all_bad20,all_bad30,all_d1,"
        << "noc_pixels,noc_valid_est_pixels,noc_valid_ratio,noc_epe,noc_bad05,noc_bad10,noc_bad20,noc_bad30,noc_d1,"
        << "occ_pixels,occ_valid_est_pixels,occ_valid_ratio,occ_epe,occ_bad05,occ_bad10,occ_bad20,occ_bad30,occ_d1,"
        << "bnd_pixels,bnd_valid_est_pixels,bnd_valid_ratio,bnd_epe,bnd_bad05,bnd_bad10,bnd_bad20,bnd_bad30,bnd_d1,"
        << "nonbnd_pixels,nonbnd_valid_est_pixels,nonbnd_valid_ratio,nonbnd_epe,nonbnd_bad05,nonbnd_bad10,nonbnd_bad20,nonbnd_bad30,nonbnd_d1,"
        << "lowtex_pixels,lowtex_valid_est_pixels,lowtex_valid_ratio,lowtex_epe,lowtex_bad05,lowtex_bad10,lowtex_bad20,lowtex_bad30,lowtex_d1,"
        << "hightex_pixels,hightex_valid_est_pixels,hightex_valid_ratio,hightex_epe,hightex_bad05,hightex_bad10,hightex_bad20,hightex_bad30,hightex_d1,"
        << "peak_workspace_bytes,retained_workspace_bytes,"
        << "avg_states_per_pixel,coarse_valid_ratio,fallback_pixel_ratio,full_range_pixel_ratio,"
        << "subpixel_method,subpixel_param,subpixel_candidate_px,subpixel_applied_px,subpixel_applied_ratio,"
        << "subpixel_degenerate_count,subpixel_clamped_count,subpixel_clamped_ratio,subpixel_neg_curv_count,"
        << "subpixel_mean_abs_delta,subpixel_p50_abs_delta,subpixel_p95_abs_delta,"
        << "inpaint_method,inpaint_gap,inpaint_total_holes,inpaint_filled_holes,inpaint_fill_ratio,"
        << "inpaint_occ_two_sided,inpaint_occ_left_only,inpaint_occ_right_only,inpaint_occ_unfilled,"
        << "inpaint_mism_two_sided,inpaint_mism_left_only,inpaint_mism_right_only,inpaint_mism_unfilled,"
        << "inpaint_reclassified_shadow,inpaint_hit_max_gap\n";

    std::cout << "Starting benchmark: " << cases.size() << " cases, "
              << ablation_ids.size() << " ablation configs, "
              << "backend_policy=" << backend_arg << ", threads=" << num_threads << " (effective=" << effective_threads << "), "
              << "warmup=" << warmup_runs << ", repeat=" << repeat_runs << "\n"
              << "Output CSV: " << output_path << "\n"
              << "------------------------------------------------------------\n";

    for (const auto& c : cases) {
        std::cout << "\n=== Dataset Case: " << c.name << " (dmax=" << c.dmax << ", dmin=" << c.dmin << ") ===\n";
        apg::Image8 left_img, right_img;
        if (!apg::load_image(c.left_path, left_img)) {
            std::cerr << "Failed to load left image: " << c.left_path << "\n";
            continue;
        }
        if (!apg::load_image(c.right_path, right_img)) {
            std::cerr << "Failed to load right image: " << c.right_path << "\n";
            continue;
        }

        apg::Image32f gt_disp;
        bool has_gt = apg::load_pfm(c.gt_path, gt_disp);
        if (!has_gt) {
            std::cout << "Note: could not load GT PFM (" << c.gt_path << "), will report metrics as N/A\n";
        }

        apg::Image8 vis_mask;
        bool has_vis = !c.vis_path.empty() && apg::load_image(c.vis_path, vis_mask);

        const int w = left_img.width();
        const int h = left_img.height();

        for (const auto& aid : ablation_ids) {
            apg::PipelineConfig cfg = apg::make_ablation_config(aid, c.dmax);
            cfg.min_disparity = c.dmin;
            cfg.max_disparity = c.dmax;
            cfg.volume_backend = volume_backend;
            if (num_threads > 0) {
                cfg.num_threads = num_threads;
            }

            if (hierarchical_enable) {
                cfg.hierarchical.enable = true;
                cfg.hierarchical.radius = hierarchical_radius;
                cfg.hierarchical.fallback = hierarchical_fallback;
                cfg.hierarchical.coarse_cross_agg = hierarchical_coarse_cross;
                cfg.hierarchical.coarse_fill_holes = hierarchical_coarse_fill;
                cfg.hierarchical.adaptive_radius = hierarchical_adaptive;
                cfg.hierarchical.confidence_threshold = hierarchical_confidence_threshold;
                cfg.hierarchical.expanded_radius = hierarchical_expanded_radius;
            }

            cfg.post.max_fill_gap = inpaint_gap;

            const bool use_packed = cfg.resolves_to_packed();
            const std::string backend_str = use_packed ? "packed" : "dense";

            apg::StereoMatcher matcher(cfg);

            // Reused O(1) buffer across runs
            apg::PipelineBuffers bufs;

            apg::ExperimentalHierarchicalScope hscope(cfg.hierarchical);
            apg::ExperimentalSubpixelScope sp_scope(subpixel_method, subpixel_param, subpixel_telem);
            apg::ExperimentalInpaintScope ip_scope(inpaint_method, inpaint_gap, inpaint_telem);

            // Warmup runs
            for (int r = 0; r < warmup_runs; ++r) {
                if (!matcher.compute(left_img, right_img, bufs, nullptr)) {
                    std::cerr << "StereoMatcher::compute failed during warmup: " << matcher.last_error() << "\n";
                    return 1;
                }
            }

            if (subpixel_telem) {
                apg::ExperimentalSubpixelScope::current_telemetry().reset();
            }
            if (inpaint_telem) {
                apg::ExperimentalInpaintScope::current_telemetry().reset();
            }

            // Timed runs
            struct TimedSample {
                double total_ms = 0.0;
                apg::PipelineStats stats;
            };
            std::vector<TimedSample> samples(repeat_runs);

            for (int r = 0; r < repeat_runs; ++r) {
                auto t0 = std::chrono::high_resolution_clock::now();
                if (!matcher.compute(left_img, right_img, bufs, &samples[r].stats)) {
                    std::cerr << "StereoMatcher::compute failed during benchmark: " << matcher.last_error() << "\n";
                    return 1;
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                samples[r].total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            // Pick median run based on total_ms
            std::sort(samples.begin(), samples.end(), [](const TimedSample& a, const TimedSample& b) {
                return a.total_ms < b.total_ms;
            });
            const auto& best_sample = samples[repeat_runs / 2];

            apg::HierarchicalTelemetry best_telem;
            if (hierarchical_enable) {
                best_telem = apg::ExperimentalHierarchicalScope::current_telemetry();
            } else {
                best_telem.avg_states_per_pixel = best_sample.stats.mean_search_width;
                best_telem.coarse_valid_ratio = 0.0f;
                best_telem.fallback_pixel_ratio = 0.0f;
                best_telem.full_range_pixel_ratio = 0.0f;
            }

            apg::SubpixelTelemetry sp_telem;
            if (subpixel_telem) {
                sp_telem = apg::ExperimentalSubpixelScope::current_telemetry();
                if (repeat_runs > 1) {
                    sp_telem.candidate_pixels /= repeat_runs;
                    sp_telem.applied_pixels /= repeat_runs;
                    sp_telem.degenerate_count /= repeat_runs;
                    sp_telem.clamped_count /= repeat_runs;
                    sp_telem.negative_curvature_count /= repeat_runs;
                    sp_telem.sum_abs_delta /= repeat_runs;
                    for (int bi = 0; bi < 50; ++bi) {
                        sp_telem.delta_buckets[bi] /= repeat_runs;
                    }
                }
            }

            // Evaluate metrics
            apg::StereoMetrics m;
            if (has_gt) {
                const apg::SearchRange* range_ptr = (cfg.prior.enable || hierarchical_enable) ? &bufs.range : nullptr;
                const apg::Image32f* before_refine_ptr = cfg.refine.enable ? &bufs.d_before_refine : nullptr;
                const apg::Image32f* after_refine_ptr = cfg.refine.enable ? &bufs.d_after_refine : nullptr;
                const apg::Image8* vis_ptr = has_vis ? &vis_mask : nullptr;
                const std::vector<apg::SupportMatch>* supp_ptr = cfg.prior.enable ? &bufs.supports : nullptr;
                m = apg::evaluate_stereo(bufs.disparity, gt_disp, range_ptr, before_refine_ptr, after_refine_ptr, vis_ptr, 1e5f, supp_ptr, &left_img);
            }

            std::string id_str = apg::ablation_id_to_string(aid);
            std::string desc_str = apg::ablation_id_description(aid);
            std::string cost_str = (cfg.cost.use_ad || cfg.cost.use_grad) ? "Census+AD+Grad" : "Census";
            std::string cross_str = cfg.aggregation.enable ? ("On(" + std::to_string(cfg.aggregation.iterations) + ")") : "Off";
            std::string p2_str = cfg.sgm.adaptive_penalties ? "Adaptive" : "Fixed";
            std::string prior_str = cfg.prior.enable ? "On" : "Off";
            std::string refine_str = cfg.refine.enable ? "On" : "Off";
            int paths = (cfg.sgm.paths == apg::PathType::Path8) ? 8 : 4;

            // Output to console
            std::cout << "[" << id_str << "] " << std::setw(32) << std::left << desc_str
                      << " | EPE: " << std::setw(6) << std::fixed << std::setprecision(3) << (has_gt ? m.epe : -1.f)
                      << " | Bad2.0: " << std::setw(5) << std::setprecision(1) << (has_gt ? m.bad_2_0 : -1.f) << "%"
                      << " | D1-all: " << std::setw(5) << std::setprecision(1) << (has_gt ? m.kitti_d1_all : -1.f) << "%";
            if (has_gt) {
                std::cout << " [Noc: " << std::setw(4) << std::setprecision(1) << m.reg_noc.bad_2_0 << "%"
                          << ", Occ: " << std::setw(4) << std::setprecision(1) << m.reg_occ.bad_2_0 << "%"
                          << ", Bnd: " << std::setw(4) << std::setprecision(1) << m.reg_boundary.bad_2_0 << "%"
                          << ", LowTex: " << std::setw(4) << std::setprecision(1) << m.reg_low_texture.bad_2_0 << "%]";
            }
            std::cout << " | D_bar: " << std::setw(4) << std::setprecision(1) << best_sample.stats.mean_search_width;
            if (cfg.prior.enable && has_gt && m.has_range) {
                std::cout << " | Rec(vis): " << std::setw(5) << std::setprecision(1) << (m.range_recall_visible * 100.0f) << "%";
                if (m.has_supports) {
                    std::cout << " [P@1(vis): " << std::setw(4) << std::setprecision(1) << (m.support_metrics.precision_vis_1 * 100.0f) << "%"
                              << " Rec(grid): " << std::setw(4) << std::setprecision(1) << (m.support_metrics.grid_recall_1 * 100.0f) << "%]";
                }
            } else {
                std::cout << " | Recall: N/A        ";
            }
            std::cout << " | Time: " << std::setw(6) << std::setprecision(2) << best_sample.total_ms << " ms"
                      << " (SGM: " << std::setprecision(2) << best_sample.stats.timing.sgm_ms << " ms)"
                      << std::endl;

            if (hierarchical_enable) {
                std::cout << "  [Hierarchical R=" << hierarchical_radius
                          << ", Fallback=" << apg::hierarchical_fallback_to_string(hierarchical_fallback)
                          << " | AvgStates: " << std::fixed << std::setprecision(1) << best_telem.avg_states_per_pixel
                          << " | CoarseValid: " << std::setprecision(1) << (best_telem.coarse_valid_ratio * 100.f) << "%"
                          << " | Fallback: " << std::setprecision(1) << (best_telem.fallback_pixel_ratio * 100.f) << "%"
                          << " | FullRange: " << std::setprecision(1) << (best_telem.full_range_pixel_ratio * 100.f) << "%"
                          << " | CoarseTime: " << std::setprecision(2) << best_telem.coarse_time_ms << " ms]\n";
            }

            if (subpixel_telem) {
                std::cout << "  [Subpixel " << apg::subpixel_method_to_string(subpixel_method)
                          << "(param=" << subpixel_param << ")"
                          << " | Cand: " << sp_telem.candidate_pixels
                          << " | App: " << sp_telem.applied_pixels << " (" << std::fixed << std::setprecision(1) << (sp_telem.applied_ratio() * 100.0) << "%)"
                          << " | Clamp: " << sp_telem.clamped_count << " (" << std::setprecision(1) << (sp_telem.clamped_ratio() * 100.0) << "%)"
                          << " | Degen: " << sp_telem.degenerate_count
                          << " | NegCurv: " << sp_telem.negative_curvature_count
                          << " | Mean|d|: " << std::setprecision(3) << sp_telem.mean_abs_delta()
                          << " | p50: " << std::setprecision(3) << sp_telem.p50_abs_delta()
                          << " | p95: " << std::setprecision(3) << sp_telem.p95_abs_delta()
                          << "]\n";
            }

            apg::InpaintTelemetry ip_telem;
            if (inpaint_telem) {
                ip_telem = apg::ExperimentalInpaintScope::current_telemetry();
                if (repeat_runs > 1) {
                    ip_telem.total_invalid_pixels /= repeat_runs;
                    ip_telem.classified_occ_pixels /= repeat_runs;
                    ip_telem.classified_mismatch_pixels /= repeat_runs;
                    ip_telem.hit_max_gap_pixels /= repeat_runs;
                    ip_telem.reclassified_shadow_pixels /= repeat_runs;
                    for (int ri = 0; ri < 11; ++ri) {
                        ip_telem.route_counts[ri] /= repeat_runs;
                    }
                    ip_telem.invalid_runs_count /= repeat_runs;
                    ip_telem.invalid_runs_truncated /= repeat_runs;
                    for (int bi = 0; bi < 10; ++bi) {
                        ip_telem.run_length_buckets[bi] /= repeat_runs;
                    }
                }
            }

            if (inpaint_telem) {
                std::cout << "  [Inpaint " << apg::inpaint_method_to_string(inpaint_method)
                          << "(gap=" << inpaint_gap << ")"
                          << " | Holes: " << ip_telem.total_invalid_pixels
                          << " | Filled: " << ip_telem.total_filled_pixels() << " (" << std::fixed << std::setprecision(1) << (ip_telem.fill_ratio() * 100.0) << "%)"
                          << " | Occ: " << ip_telem.classified_occ_pixels << " (Both: " << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::OccBothMin)]
                          << ", L: " << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::OccLeftOnly)]
                          << ", R: " << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::OccRightOnly)] << ")"
                          << " | Mism: " << ip_telem.classified_mismatch_pixels << " (Lerp: " << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::MismatchLerp)] << ")"
                          << " | HitGap: " << ip_telem.hit_max_gap_pixels
                          << " | ReclassShadow: " << ip_telem.reclassified_shadow_pixels
                          << "]\n";
            }

            // Output to CSV
            csv << backend_str << ","
                << num_threads << ","
                << effective_threads << ","
                << warmup_runs << ","
                << repeat_runs << ","
                << build_type << ","
                << c.name << ","
                << id_str << ",\""
                << desc_str << "\","
                << cost_str << ","
                << cross_str << ","
                << p2_str << ","
                << prior_str << ","
                << refine_str << ","
                << paths << ","
                << w << ","
                << h << ","
                << c.dmax << ","
                << std::fixed << std::setprecision(4)
                << (has_gt ? m.epe : 0.f) << ","
                << (has_gt ? m.bad_0_5 : 0.f) << ","
                << (has_gt ? m.bad_1_0 : 0.f) << ","
                << (has_gt ? m.bad_2_0 : 0.f) << ","
                << (has_gt ? m.bad_3_0 : 0.f) << ","
                << (has_gt ? m.kitti_d1_all : 0.f) << ","
                << (has_gt ? m.kitti_d1_noc : 0.f) << ","
                << (has_gt ? m.valid_ratio : 0.f) << ","
                << (has_gt ? m.lr_fail_ratio : 0.f) << ","
                << (has_gt ? m.edge_epe : 0.f) << ","
                << (has_gt ? m.nonedge_epe : 0.f) << ",";

            if (cfg.prior.enable && has_gt && m.has_range) {
                csv << std::fixed << std::setprecision(4)
                    << m.range_recall_all << ","
                    << m.range_recall_matchable << ","
                    << m.range_recall_visible << ","
                    << m.range_visible_eval_pixels << ","
                    << m.range_visible_in_pixels << ","
                    << m.range_matchable_eval_pixels << ","
                    << m.range_matchable_in_pixels << ","
                    << m.prior_miss_visible_count << ","
                    << std::setprecision(2)
                    << (m.prior_miss_edge_ratio * 100.0f) << ","
                    << (m.prior_miss_nonedge_ratio * 100.0f) << ",";
            } else {
                csv << "N/A,N/A,N/A,0,0,0,0,0,0.00,0.00,";
            }

            if (cfg.prior.enable && has_gt && m.has_supports) {
                csv << best_sample.stats.prior_support_count << ","
                    << std::setprecision(4)
                    << m.support_metrics.precision_all_05 << ","
                    << m.support_metrics.precision_all_1 << ","
                    << m.support_metrics.precision_all_2 << ","
                    << m.support_metrics.precision_vis_1 << ","
                    << m.support_metrics.precision_vis_05 << ","
                    << m.support_metrics.precision_vis_1 << ","
                    << m.support_metrics.precision_vis_2 << ","
                    << m.support_metrics.gt_valid << ","
                    << m.support_metrics.visible << ","
                    << m.support_metrics.correct_05 << ","
                    << m.support_metrics.correct_1 << ","
                    << m.support_metrics.correct_2 << ","
                    << m.support_metrics.visible_correct_05 << ","
                    << m.support_metrics.visible_correct_1 << ","
                    << m.support_metrics.visible_correct_2 << ","
                    << m.support_metrics.visible_gt_cells << ","
                    << m.support_metrics.correct_visible_cells << ","
                    << m.support_metrics.mae_all << ","
                    << m.support_metrics.mae_visible << ","
                    << m.support_metrics.grid_recall_1 << ",";
            } else {
                csv << best_sample.stats.prior_support_count << ",0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0,0,0,0,0,0,0,0.0,0.0,0.0,";
            }

            csv << std::setprecision(2)
                << best_sample.stats.mean_search_width << ","
                << best_sample.stats.mean_geometry_width << ","
                << std::setprecision(4)
                << best_sample.stats.geometry_reduction_ratio << ","
                << (cfg.prior.enable ? best_sample.stats.prior_incremental_reduction_ratio : 0.0) << ","
                << best_sample.stats.search_reduction_ratio << ","
                << best_sample.stats.cost_bytes << ","
                << best_sample.stats.aggregated_cost_bytes << ","
                << best_sample.stats.estimated_peak_bytes << ","
                << best_sample.stats.reliable_ratio << ","
                << best_sample.stats.unreliable_ratio << ","
                << (m.has_refine_stats ? m.refine_changed_ratio : 0.f) << ","
                << (m.has_refine_stats ? m.refine_epe_delta : 0.f) << ","
                << (m.has_refine_stats ? m.post_epe_delta : 0.f) << ","
                << (m.has_refine_stats ? m.total_epe_delta : 0.f) << ","
                << std::setprecision(2)
                << best_sample.total_ms << ","
                << best_sample.stats.timing.aux_ms << ","
                << best_sample.stats.timing.prior_ms << ","
                << best_sample.stats.timing.cost_ms << ","
                << best_sample.stats.timing.right_wta_ms << ","
                << best_sample.stats.timing.cross_ms << ","
                << best_sample.stats.timing.sgm_ms << ","
                << best_sample.stats.timing.wta_ms << ","
                << best_sample.stats.timing.confidence_ms << ","
                << best_sample.stats.timing.refine_ms << ","
                << best_sample.stats.timing.post_ms << ","
                << m.reg_all.total_pixels << ","
                << m.reg_all.valid_est_pixels << ","
                << std::setprecision(4)
                << m.reg_all.valid_ratio << ","
                << m.reg_all.epe << ","
                << m.reg_all.bad_0_5 << ","
                << m.reg_all.bad_1_0 << ","
                << m.reg_all.bad_2_0 << ","
                << m.reg_all.bad_3_0 << ","
                << m.reg_all.d1 << ","
                << m.reg_noc.total_pixels << ","
                << m.reg_noc.valid_est_pixels << ","
                << m.reg_noc.valid_ratio << ","
                << m.reg_noc.epe << ","
                << m.reg_noc.bad_0_5 << ","
                << m.reg_noc.bad_1_0 << ","
                << m.reg_noc.bad_2_0 << ","
                << m.reg_noc.bad_3_0 << ","
                << m.reg_noc.d1 << ","
                << m.reg_occ.total_pixels << ","
                << m.reg_occ.valid_est_pixels << ","
                << m.reg_occ.valid_ratio << ","
                << m.reg_occ.epe << ","
                << m.reg_occ.bad_0_5 << ","
                << m.reg_occ.bad_1_0 << ","
                << m.reg_occ.bad_2_0 << ","
                << m.reg_occ.bad_3_0 << ","
                << m.reg_occ.d1 << ","
                << m.reg_boundary.total_pixels << ","
                << m.reg_boundary.valid_est_pixels << ","
                << m.reg_boundary.valid_ratio << ","
                << m.reg_boundary.epe << ","
                << m.reg_boundary.bad_0_5 << ","
                << m.reg_boundary.bad_1_0 << ","
                << m.reg_boundary.bad_2_0 << ","
                << m.reg_boundary.bad_3_0 << ","
                << m.reg_boundary.d1 << ","
                << m.reg_non_boundary.total_pixels << ","
                << m.reg_non_boundary.valid_est_pixels << ","
                << m.reg_non_boundary.valid_ratio << ","
                << m.reg_non_boundary.epe << ","
                << m.reg_non_boundary.bad_0_5 << ","
                << m.reg_non_boundary.bad_1_0 << ","
                << m.reg_non_boundary.bad_2_0 << ","
                << m.reg_non_boundary.bad_3_0 << ","
                << m.reg_non_boundary.d1 << ","
                << m.reg_low_texture.total_pixels << ","
                << m.reg_low_texture.valid_est_pixels << ","
                << m.reg_low_texture.valid_ratio << ","
                << m.reg_low_texture.epe << ","
                << m.reg_low_texture.bad_0_5 << ","
                << m.reg_low_texture.bad_1_0 << ","
                << m.reg_low_texture.bad_2_0 << ","
                << m.reg_low_texture.bad_3_0 << ","
                << m.reg_low_texture.d1 << ","
                << m.reg_high_texture.total_pixels << ","
                << m.reg_high_texture.valid_est_pixels << ","
                << m.reg_high_texture.valid_ratio << ","
                << m.reg_high_texture.epe << ","
                << m.reg_high_texture.bad_0_5 << ","
                << m.reg_high_texture.bad_1_0 << ","
                << m.reg_high_texture.bad_2_0 << ","
                << m.reg_high_texture.bad_3_0 << ","
                << m.reg_high_texture.d1 << ","
                << ((best_sample.stats.peak_workspace_bytes > 0) ? best_sample.stats.peak_workspace_bytes : best_sample.stats.estimated_peak_bytes) << ","
                << best_sample.stats.retained_workspace_bytes << ","
                << std::fixed << std::setprecision(4)
                << best_telem.avg_states_per_pixel << ","
                << best_telem.coarse_valid_ratio << ","
                << best_telem.fallback_pixel_ratio << ","
                << best_telem.full_range_pixel_ratio << ","
                << apg::subpixel_method_to_string(subpixel_method) << ","
                << subpixel_param << ","
                << sp_telem.candidate_pixels << ","
                << sp_telem.applied_pixels << ","
                << sp_telem.applied_ratio() << ","
                << sp_telem.degenerate_count << ","
                << sp_telem.clamped_count << ","
                << sp_telem.clamped_ratio() << ","
                << sp_telem.negative_curvature_count << ","
                << sp_telem.mean_abs_delta() << ","
                << sp_telem.p50_abs_delta() << ","
                << sp_telem.p95_abs_delta() << ","
                << apg::inpaint_method_to_string(inpaint_method) << ","
                << inpaint_gap << ","
                << ip_telem.total_invalid_pixels << ","
                << ip_telem.total_filled_pixels() << ","
                << ip_telem.fill_ratio() << ","
                << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::OccBothMin)] << ","
                << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::OccLeftOnly)] << ","
                << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::OccRightOnly)] << ","
                << ip_telem.occ_unfilled_pixels() << ","
                << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::MismatchLerp)] << ","
                << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::MismatchLeftOnly)] << ","
                << ip_telem.route_counts[static_cast<size_t>(apg::FillRoute::MismatchRightOnly)] << ","
                << ip_telem.mismatch_unfilled_pixels() << ","
                << ip_telem.reclassified_shadow_pixels << ","
                << ip_telem.hit_max_gap_pixels << "\n";
            csv.flush();

            // Save disparity maps if requested
            if (!save_disp_dir.empty()) {
                std::string prefix = save_disp_dir + "/" + c.name + "_" + id_str;
                apg::save_pfm(prefix + "_disp.pfm", bufs.disparity);
                apg::save_disparity_preview(prefix + "_disp.pgm", bufs.disparity, static_cast<float>(c.dmax));
                if (cfg.prior.enable && has_gt && m.has_range) {
                    apg::Image8 miss_img(w, h, 1);
                    for (int y = 0; y < h; ++y) {
                        for (int x = 0; x < w; ++x) {
                            const float g = gt_disp.at(x, y);
                            if (g < 0.0f || !std::isfinite(g)) continue;
                            const int xr = x - static_cast<int>(std::round(g));
                            const bool is_matchable = (xr >= 0 && xr < w);
                            const bool is_vis = has_vis ? (vis_mask.at(x, y) >= 200) : is_matchable;
                            if (is_vis) {
                                if (g < bufs.range.dmin.at(x, y) || g >= bufs.range.dmax.at(x, y)) {
                                    miss_img.at(x, y) = 255;
                                }
                            }
                        }
                    }
                    apg::save_pgm(prefix + "_miss.pgm", miss_img);
                }
            }
        }
    }

    std::cout << "\nBenchmark complete. Results written to: " << output_path << std::endl;
    return 0;
}
