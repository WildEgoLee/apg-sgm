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
};

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

        fs::path lp(c.left_path);
        if (lp.is_relative()) c.left_path = (base_dir / lp).lexically_normal().string();

        fs::path rp(c.right_path);
        if (rp.is_relative()) c.right_path = (base_dir / rp).lexically_normal().string();

        fs::path gp(c.gt_path);
        if (gp.is_relative()) c.gt_path = (base_dir / gp).lexically_normal().string();

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

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--ablation" && i + 1 < argc) {
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: apg-benchmark [options]\n"
                      << "Options:\n"
                      << "  --manifest <path>     Path to dataset manifest file (required)\n"
                      << "  --ablation <id|all>   Ablation experiment: A, B, C, D, E, F, G, or all (default: all)\n"
                      << "  --output <path>       Output CSV file path (default: results/ablation.csv)\n"
                      << "  --save_disp <dir>     Optional directory to save output disparity maps\n"
                      << "  --warmup <N>          Warmup runs before timing (default: 2)\n"
                      << "  --repeat <N>          Timed runs per test case (default: 5, reports median)\n"
                      << "  --threads <N>         Number of OpenMP threads (default: auto)\n"
                      << "  --help, -h            Show this help message\n";
            return 0;
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

    csv << "case,ablation_id,ablation_name,cost,cross,p2,prior,refine,paths,w,h,dmax,"
        << "epe,bad_0_5,bad_1_0,bad_2_0,bad_3_0,valid_ratio,lr_fail_ratio,edge_epe,nonedge_epe,range_recall,"
        << "time_total_ms,time_cost_ms,time_cross_ms,time_sgm_ms,time_prior_ms,time_refine_ms,"
        << "refine_epe_delta,refine_improved_pct\n";

    std::cout << "Starting benchmark: " << cases.size() << " cases, "
              << ablation_ids.size() << " ablation configs, "
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

        const int w = left_img.width();
        const int h = left_img.height();

        for (const auto& aid : ablation_ids) {
            apg::PipelineConfig cfg = apg::make_ablation_config(aid, c.dmax);
            cfg.min_disparity = c.dmin;
            cfg.max_disparity = c.dmax;
            if (num_threads > 0) {
                cfg.num_threads = num_threads;
            }

            apg::StereoMatcher matcher(cfg);

            // Warmup runs
            apg::PipelineBuffers warmup_bufs;
            for (int r = 0; r < warmup_runs; ++r) {
                matcher.compute(left_img, right_img, warmup_bufs, nullptr);
            }

            // Timed runs
            struct TimedSample {
                double total_ms = 0.0;
                apg::PipelineStats stats;
                apg::PipelineBuffers bufs;
            };
            std::vector<TimedSample> samples(repeat_runs);

            for (int r = 0; r < repeat_runs; ++r) {
                auto t0 = std::chrono::high_resolution_clock::now();
                matcher.compute(left_img, right_img, samples[r].bufs, &samples[r].stats);
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
                const apg::SearchRange* range_ptr = cfg.prior.enable ? &best_sample.bufs.range : nullptr;
                const apg::Image32f* before_refine_ptr = cfg.refine.enable ? &best_sample.bufs.d_before_refine : nullptr;
                m = apg::evaluate_stereo(best_sample.bufs.disparity, gt_disp, range_ptr, before_refine_ptr, static_cast<float>(c.dmax));
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
                      << " | Time: " << std::setw(6) << std::setprecision(2) << best_sample.total_ms << " ms"
                      << " (SGM: " << std::setprecision(2) << best_sample.stats.timing.sgm_ms << " ms)"
                      << std::endl;

            // Output to CSV
            csv << c.name << ","
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
                << (has_gt ? m.valid_ratio : 0.f) << ","
                << (has_gt ? m.lr_fail_ratio : 0.f) << ","
                << (has_gt ? m.edge_epe : 0.f) << ","
                << (has_gt ? m.nonedge_epe : 0.f) << ","
                << (has_gt ? m.range_gt_recall : 0.f) << ","
                << std::setprecision(2)
                << best_sample.total_ms << ","
                << best_sample.stats.timing.cost_ms << ","
                << best_sample.stats.timing.cross_ms << ","
                << best_sample.stats.timing.sgm_ms << ","
                << best_sample.stats.timing.prior_ms << ","
                << best_sample.stats.timing.refine_ms << ","
                << std::setprecision(4)
                << (m.has_refine_stats ? m.refine_epe_delta : 0.f) << ","
                << (m.has_refine_stats ? m.refine_improved_ratio * 100.0f : 0.f) << "\n";
            csv.flush();

            // Save disparity maps if requested
            if (!save_disp_dir.empty()) {
                std::string prefix = save_disp_dir + "/" + c.name + "_" + id_str;
                apg::save_pfm(prefix + "_disp.pfm", best_sample.bufs.disparity);
                apg::save_disparity_preview(prefix + "_disp.pgm", best_sample.bufs.disparity, static_cast<float>(c.dmax));
            }
        }
    }

    std::cout << "\nBenchmark complete. Results written to: " << output_path << std::endl;
    return 0;
}
