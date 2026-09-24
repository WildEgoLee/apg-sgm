#include "apg_sgm/ablation.hpp"
#include "apg_sgm/image.hpp"
#include "apg_sgm/metrics.hpp"
#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/pipeline_stats.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
                diff = (v1 != v2 && !(std::isnan(v1) && std::isnan(v2)));
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
    bool use_packed = false;
    bool verify_backends = false;

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
        } else if (arg == "--packed") {
            use_packed = true;
        } else if (arg == "--verify-backends") {
            verify_backends = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: apg-benchmark [options]\n"
                      << "Options:\n"
                      << "  --manifest <path>     Path to dataset manifest file (required)\n"
                      << "  --ablation, --modes <id|all>\n"
                      << "                        Ablation experiment: A, B, C, D, E, F, G, or all (default: all)\n"
                      << "  --output <path>       Output CSV file path (default: results/ablation.csv)\n"
                      << "  --save_disp <dir>     Optional directory to save output disparity maps\n"
                      << "  --warmup <N>          Warmup runs before timing (default: 2)\n"
                      << "  --repeat <N>          Timed runs per test case (default: 5, reports median)\n"
                      << "  --threads <N>         Number of OpenMP threads (default: 0 = auto)\n"
                      << "  --packed              Use packed cost volume backend (Sigma_p D(p))\n"
                      << "  --verify-backends     Verify 100% bit-exact parity between dense and packed backends\n"
                      << "  --help, -h            Show this help message\n";
            return 0;
        } else {
            std::cerr << "Error: unknown or incomplete argument '" << arg << "'. Use --help for usage.\n";
            return 1;
        }
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
                cfg_dense.use_packed_volume = false;
                if (num_threads > 0) cfg_dense.num_threads = num_threads;

                apg::PipelineConfig cfg_packed = cfg_dense;
                cfg_packed.use_packed_volume = true;

                apg::StereoMatcher matcher_dense(cfg_dense);
                apg::StereoMatcher matcher_packed(cfg_packed);

                apg::PipelineBuffers bufs_dense;
                apg::PipelineBuffers bufs_packed;

                matcher_dense.compute(left_img, right_img, bufs_dense, nullptr);
                matcher_packed.compute(left_img, right_img, bufs_packed, nullptr);

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
    const std::string backend_str = use_packed ? "packed" : "dense";

    csv << "backend,threads,warmup,repeat,build_type,"
        << "case,ablation_id,ablation_name,cost,cross,p2,prior,refine,paths,w,h,dmax,"
        << "epe,bad_0_5,bad_1_0,bad_2_0,bad_3_0,kitti_d1_all,kitti_d1_noc,valid_ratio,lr_fail_ratio,edge_epe,nonedge_epe,"
        << "recall_all,recall_matchable,recall_visible,range_vis_eval_px,range_vis_in_px,range_mat_eval_px,range_mat_in_px,"
        << "prior_miss_vis_count,prior_miss_edge_pct,prior_miss_nonedge_pct,"
        << "prior_supports,support_p05,support_p1,support_p2,support_p1_vis,support_mae,support_mae_vis,grid_recall_1,"
        << "mean_search_width,mean_geom_width,geom_reduction_ratio,prior_incremental_reduction,total_reduction_ratio,"
        << "cost_bytes,cost32_bytes,peak_bytes,"
        << "reliable_ratio,unreliable_ratio,refine_changed_ratio,refine_epe_delta,post_epe_delta,total_epe_delta,"
        << "time_total_ms,time_cost_ms,time_right_wta_ms,time_cross_ms,time_sgm_ms,time_prior_ms,time_refine_ms,time_post_ms\n";

    std::cout << "Starting benchmark: " << cases.size() << " cases, "
              << ablation_ids.size() << " ablation configs, "
              << "backend=" << backend_str << ", threads=" << num_threads << ", "
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
            cfg.use_packed_volume = use_packed;
            if (num_threads > 0) {
                cfg.num_threads = num_threads;
            }

            apg::StereoMatcher matcher(cfg);

            // Reused O(1) buffer across runs
            apg::PipelineBuffers bufs;

            // Warmup runs
            for (int r = 0; r < warmup_runs; ++r) {
                matcher.compute(left_img, right_img, bufs, nullptr);
            }

            // Timed runs
            struct TimedSample {
                double total_ms = 0.0;
                apg::PipelineStats stats;
            };
            std::vector<TimedSample> samples(repeat_runs);

            for (int r = 0; r < repeat_runs; ++r) {
                auto t0 = std::chrono::high_resolution_clock::now();
                matcher.compute(left_img, right_img, bufs, &samples[r].stats);
                auto t1 = std::chrono::high_resolution_clock::now();
                samples[r].total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            // Pick median run based on total_ms
            std::sort(samples.begin(), samples.end(), [](const TimedSample& a, const TimedSample& b) {
                return a.total_ms < b.total_ms;
            });
            const auto& best_sample = samples[repeat_runs / 2];

            // Evaluate metrics
            apg::StereoMetrics m;
            if (has_gt) {
                const apg::SearchRange* range_ptr = cfg.prior.enable ? &bufs.range : nullptr;
                const apg::Image32f* before_refine_ptr = cfg.refine.enable ? &bufs.d_before_refine : nullptr;
                const apg::Image32f* after_refine_ptr = cfg.refine.enable ? &bufs.d_after_refine : nullptr;
                const apg::Image8* vis_ptr = has_vis ? &vis_mask : nullptr;
                const std::vector<apg::SupportMatch>* supp_ptr = cfg.prior.enable ? &bufs.supports : nullptr;
                m = apg::evaluate_stereo(bufs.disparity, gt_disp, range_ptr, before_refine_ptr, after_refine_ptr, vis_ptr, static_cast<float>(c.dmax), supp_ptr);
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
                      << " | D1-all: " << std::setw(5) << std::setprecision(1) << (has_gt ? m.kitti_d1_all : -1.f) << "%"
                      << " | D_bar: " << std::setw(4) << std::setprecision(1) << best_sample.stats.mean_search_width;
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

            // Output to CSV
            csv << backend_str << ","
                << num_threads << ","
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
                    << m.support_metrics.mae_all << ","
                    << m.support_metrics.mae_visible << ","
                    << m.support_metrics.grid_recall_1 << ",";
            } else {
                csv << best_sample.stats.prior_support_count << ",0.0,0.0,0.0,0.0,0.0,0.0,0.0,";
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
                << best_sample.stats.timing.cost_ms << ","
                << best_sample.stats.timing.right_wta_ms << ","
                << best_sample.stats.timing.cross_ms << ","
                << best_sample.stats.timing.sgm_ms << ","
                << best_sample.stats.timing.prior_ms << ","
                << best_sample.stats.timing.refine_ms << ","
                << best_sample.stats.timing.post_ms << "\n";
            csv.flush();

            // Save disparity maps if requested
            if (!save_disp_dir.empty()) {
                std::string prefix = save_disp_dir + "/" + c.name + "_" + id_str;
                apg::save_pfm(prefix + "_disp.pfm", bufs.disparity);
                apg::save_disparity_preview(prefix + "_disp.pgm", bufs.disparity, static_cast<float>(c.dmax));
            }
        }
    }

    std::cout << "\nBenchmark complete. Results written to: " << output_path << std::endl;
    return 0;
}
