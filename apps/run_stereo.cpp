#include "apg_sgm/pipeline.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

using namespace apg;

static void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0
        << " left.pgm right.pgm out.pgm [--mode fast|balanced|high] [--disp-max N] [--threads N]\n"
        << "Inputs: binary PGM (P5) or PPM (P6). Images must already be rectified.\n";
}

static QualityMode parse_mode(const std::string& s) {
    if (s == "fast") return QualityMode::Fast;
    if (s == "high" || s == "hq" || s == "highquality") return QualityMode::HighQuality;
    return QualityMode::Balanced;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    const std::string left_path = argv[1];
    const std::string right_path = argv[2];
    const std::string out_path = argv[3];

    QualityMode mode = QualityMode::Balanced;
    int dmax = 128;
    int threads = 0;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = parse_mode(argv[++i]);
        } else if (std::strcmp(argv[i], "--disp-max") == 0 && i + 1 < argc) {
            dmax = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    Image8 left, right;
    if (!load_image(left_path, left)) {
        std::cerr << "failed to load " << left_path << "\n";
        return 1;
    }
    if (!load_image(right_path, right)) {
        std::cerr << "failed to load " << right_path << "\n";
        return 1;
    }

    PipelineConfig cfg = PipelineConfig::from_mode(mode, dmax);
    cfg.num_threads = threads;
    StereoMatcher matcher(cfg);
    PipelineBuffers buf;

    const auto t0 = std::chrono::steady_clock::now();
    if (!matcher.compute(left, right, buf)) {
        std::cerr << "compute failed: " << matcher.last_error() << "\n";
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!save_disparity_preview(out_path, buf.disparity, static_cast<float>(dmax))) {
        std::cerr << "failed to write " << out_path << "\n";
        return 1;
    }

    const std::string conf_path = out_path + ".conf.pgm";
    save_disparity_preview(conf_path, buf.confidence, 2.f);

    std::cout << "ok  " << left.width() << "x" << left.height()
              << "  D=[" << cfg.min_disparity << "," << cfg.max_disparity << ")"
              << "  " << ms << " ms\n"
              << "disparity preview: " << out_path << "\n"
              << "confidence preview: " << conf_path << "\n";
    return 0;
}
