#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/status.hpp"
#include "apg_sgm/image.hpp"

#include <iostream>
#include <cassert>

using namespace apg;

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "               TEST STATUS & STRUCTURED VALIDATION ARCHITECTURE          " << std::endl;
    std::cout << "==========================================================================" << std::endl;

    // 1. Test validate_inputs
    {
        Image8 empty_img;
        Image8 valid_img(16, 16, 1);
        Image8 diff_size(16, 18, 1);
        Image8 diff_channel(16, 16, 3);

        Status s1 = validate_inputs(empty_img, valid_img);
        if (s1.code != StatusCode::EmptyInput) {
            std::cerr << "Expected EmptyInput, got: " << status_code_string(s1.code) << std::endl;
            return 1;
        }

        Status s2 = validate_inputs(valid_img, diff_size);
        if (s2.code != StatusCode::SizeMismatch) {
            std::cerr << "Expected SizeMismatch, got: " << status_code_string(s2.code) << std::endl;
            return 1;
        }

        Status s3 = validate_inputs(valid_img, diff_channel);
        if (s3.code != StatusCode::SizeMismatch) {
            std::cerr << "Expected SizeMismatch for channel count, got: " << status_code_string(s3.code) << std::endl;
            return 1;
        }

        Status s4 = validate_inputs(valid_img, valid_img);
        if (!s4.ok()) {
            std::cerr << "Expected Ok, got: " << status_code_string(s4.code) << std::endl;
            return 1;
        }
        std::cout << "  Input validation tests: PASS\n";
    }

    // 2. Test validate_config
    {
        PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced);

        // Invalid disparity range
        cfg.min_disparity = 32;
        cfg.max_disparity = 16;
        Status s1 = validate_config(cfg);
        if (s1.code != StatusCode::InvalidDisparityRange) {
            std::cerr << "Expected InvalidDisparityRange, got: " << status_code_string(s1.code) << std::endl;
            return 1;
        }

        // Invalid P2_base <= P1
        cfg = PipelineConfig::from_mode(QualityMode::Balanced);
        cfg.sgm.P1 = 50;
        cfg.sgm.P2_base = 50;
        Status s2 = validate_config(cfg);
        if (s2.code != StatusCode::InvalidConfig) {
            std::cerr << "Expected InvalidConfig for P2 <= P1, got: " << status_code_string(s2.code) << std::endl;
            return 1;
        }

        // Invalid gradient thresholds
        cfg = PipelineConfig::from_mode(QualityMode::Balanced);
        cfg.sgm.grad_t1 = 30;
        cfg.sgm.grad_t2 = 10;
        Status s3 = validate_config(cfg);
        if (s3.code != StatusCode::InvalidConfig) {
            std::cerr << "Expected InvalidConfig for grad_t1 > grad_t2, got: " << status_code_string(s3.code) << std::endl;
            return 1;
        }

        // Valid config
        cfg = PipelineConfig::from_mode(QualityMode::Balanced);
        Status s4 = validate_config(cfg);
        if (!s4.ok()) {
            std::cerr << "Expected Ok for default config, got: " << status_code_string(s4.code) << std::endl;
            return 1;
        }
        std::cout << "  Config validation tests: PASS\n";
    }

    // 3. Test StereoMatcher integration with Status API & Legacy last_error API
    {
        PipelineConfig cfg = PipelineConfig::from_mode(QualityMode::Balanced);
        cfg.min_disparity = 64;
        cfg.max_disparity = 32;
        StereoMatcher matcher(cfg);

        Image8 left(16, 16, 1);
        Image8 right(16, 16, 1);
        StereoResult res;

        // compute_status returns structured Status
        Status s = matcher.compute_status(left, right, res);
        if (s.code != StatusCode::InvalidDisparityRange) {
            std::cerr << "Expected compute_status to return InvalidDisparityRange, got: " << status_code_string(s.code) << std::endl;
            return 1;
        }

        // compute returns bool and updates last_error_
        bool ok = matcher.compute(left, right, res);
        if (ok) {
            std::cerr << "Expected compute to return false on invalid config\n";
            return 1;
        }
        if (matcher.last_error().empty()) {
            std::cerr << "Expected matcher.last_error() to be populated on failure\n";
            return 1;
        }
        std::cout << "  StereoMatcher Status & Legacy compatibility tests: PASS\n";
    }

    std::cout << "All Status & validation tests passed!\n";
    return 0;
}
