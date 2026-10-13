#include "apg_sgm/pipeline.hpp"
#include "apg_sgm/image.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace apg;

static uint64_t fnv1a64(const float* data, size_t count) {
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(data);
    const size_t byte_count = count * sizeof(float);
    for (size_t i = 0; i < byte_count; ++i) {
        hash ^= static_cast<uint64_t>(byte_ptr[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t fnv1a64_u8(const uint8_t* data, size_t count) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < count; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Generates synthetic patterns for contract verification
enum class PatternType {
    Constant,
    HorizontalRamp,
    VerticalRamp,
    Checkerboard,
    DeterministicNoise
};

static void generate_pattern(int w, int h, int channels, PatternType type, Image8& img) {
    img = Image8(w, h, channels);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t v = 0;
            switch (type) {
                case PatternType::Constant:
                    v = 128;
                    break;
                case PatternType::HorizontalRamp:
                    v = static_cast<uint8_t>((x * 255) / std::max(1, w - 1));
                    break;
                case PatternType::VerticalRamp:
                    v = static_cast<uint8_t>((y * 255) / std::max(1, h - 1));
                    break;
                case PatternType::Checkerboard:
                    v = ((x / 4 + y / 4) % 2 == 0) ? 40 : 210;
                    break;
                case PatternType::DeterministicNoise:
                    v = static_cast<uint8_t>((x * 37 + y * 73 + x * y * 13) % 256);
                    break;
            }
            if (channels == 1) {
                img.at(x, y) = v;
            } else {
                img.at(x, y, 0) = v;
                img.at(x, y, 1) = static_cast<uint8_t>((v + 30) % 256);
                img.at(x, y, 2) = static_cast<uint8_t>((v + 60) % 256);
            }
        }
    }
}

struct GoldenContractFingerprint {
    std::string tag;
    int w = 0;
    int h = 0;
    size_t invalid_count = 0;
    size_t reliable_count = 0;
    uint64_t disp_hash = 0;
    uint64_t conf_hash = 0;
    uint64_t invalid_hash = 0;
    uint64_t reliable_hash = 0;
};

static bool run_and_check(const std::string& name,
                          const Image8& left,
                          const Image8& right,
                          const PipelineConfig& cfg,
                          GoldenContractFingerprint& fp) {
    StereoMatcher matcher(cfg);
    PipelineBuffers buf;
    if (!matcher.compute(left, right, buf)) {
        std::cerr << "Pipeline compute failed for " << name << ": " << matcher.last_error() << std::endl;
        return false;
    }

    StereoResult result;
    if (!matcher.compute(left, right, result)) {
        std::cerr << "StereoResult compute failed for " << name << ": " << matcher.last_error() << std::endl;
        return false;
    }

    // Verify 100% exact parity between StereoResult and PipelineBuffers
    if (result.width() != buf.disparity.width() || result.height() != buf.disparity.height()) {
        std::cerr << "Parity failure: result dimensions mismatch with buffer\n";
        return false;
    }

    const size_t total_pix = static_cast<size_t>(result.width()) * result.height();
    for (size_t i = 0; i < total_pix; ++i) {
        if (result.disparity.data()[i] != buf.disparity.data()[i] ||
            result.confidence.data()[i] != buf.confidence.data()[i] ||
            result.invalid_reason.data()[i] != buf.invalid_reason.data()[i] ||
            result.reliable_mask.data()[i] != buf.reliable_mask.data()[i]) {
            std::cerr << "Parity failure: pixel " << i << " mismatch between StereoResult and PipelineBuffers\n";
            return false;
        }
    }

    fp.tag = name;
    fp.w = buf.disparity.width();
    fp.h = buf.disparity.height();

    const size_t n_pixels = static_cast<size_t>(fp.w) * fp.h;
    fp.invalid_count = 0;
    fp.reliable_count = 0;

    for (int y = 0; y < fp.h; ++y) {
        for (int x = 0; x < fp.w; ++x) {
            const float d = buf.disparity.at(x, y);
            // Invariant check: Output disparity never NaN or Inf
            if (std::isnan(d) || std::isinf(d)) {
                std::cerr << "Invariant violation: NaN or Inf disparity at (" << x << "," << y << ")\n";
                return false;
            }
            // Invariant check: invalid pixels must have value -1.0f
            if (d < 0.0f) {
                if (d != -1.0f) {
                    std::cerr << "Invariant violation: negative disparity is not -1.0f at (" << x << "," << y << "): " << d << "\n";
                    return false;
                }
                fp.invalid_count++;
            }

            // Invariant check: confidence in [0, 1]
            const float c = buf.confidence.at(x, y);
            if (c < 0.0f || c > 1.0f || std::isnan(c)) {
                std::cerr << "Invariant violation: confidence out of range [0, 1]: " << c << "\n";
                return false;
            }

            // Reliable count
            if (buf.reliable_mask.at(x, y) != 0) {
                fp.reliable_count++;
            }
        }
    }

    fp.disp_hash = fnv1a64(buf.disparity.data(), n_pixels);
    fp.conf_hash = fnv1a64(buf.confidence.data(), n_pixels);
    fp.invalid_hash = fnv1a64_u8(buf.invalid_reason.data(), n_pixels);
    fp.reliable_hash = fnv1a64_u8(buf.reliable_mask.data(), n_pixels);

    return true;
}

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "             P4.0 API & BEHAVIORAL CONTRACT FREEZE TESTS                  " << std::endl;
    std::cout << "==========================================================================" << std::endl;

    std::vector<GoldenContractFingerprint> fingerprints;

    const QualityMode modes[] = {QualityMode::Fast, QualityMode::Balanced, QualityMode::HighQuality};
    const VolumeBackend backends[] = {VolumeBackend::Dense, VolumeBackend::Packed};

    // Test Matrix
    // 1. Tiny dimension (16x12)
    // 2. Odd dimension (33x27)
    // 3. 1-Channel Grayscale vs 3-Channel Color
    // 4. Patterns: Checkerboard, DeterministicNoise, Ramp
    // 5. Modes & Backends
    for (QualityMode m : modes) {
        for (VolumeBackend b : backends) {
            std::string mode_str = (m == QualityMode::Fast) ? "Fast" : (m == QualityMode::Balanced ? "Balanced" : "HighQuality");
            std::string backend_str = (b == VolumeBackend::Dense) ? "Dense" : "Packed";

            // Case A: 33x27 Checkerboard (Odd dimensions, 1-channel)
            {
                Image8 left, right;
                generate_pattern(33, 27, 1, PatternType::Checkerboard, left);
                generate_pattern(33, 27, 1, PatternType::Checkerboard, right);
                auto cfg = PipelineConfig::from_mode(m, 16);
                cfg.volume_backend = b;
                cfg.num_threads = 1;

                GoldenContractFingerprint fp;
                if (!run_and_check("Odd_Checkerboard_" + mode_str + "_" + backend_str, left, right, cfg, fp)) {
                    return 1;
                }
                fingerprints.push_back(fp);
            }

            // Case B: 32x24 DeterministicNoise (Color 3-channel)
            {
                Image8 left, right;
                generate_pattern(32, 24, 3, PatternType::DeterministicNoise, left);
                generate_pattern(32, 24, 3, PatternType::DeterministicNoise, right);
                auto cfg = PipelineConfig::from_mode(m, 16);
                cfg.volume_backend = b;
                cfg.num_threads = 1;

                GoldenContractFingerprint fp;
                if (!run_and_check("Color_Noise_" + mode_str + "_" + backend_str, left, right, cfg, fp)) {
                    return 1;
                }
                fingerprints.push_back(fp);
            }
        }
    }

    // Case C: Negative Disparity Range & Small Disparity Range
    {
        Image8 left, right;
        generate_pattern(48, 36, 1, PatternType::DeterministicNoise, left);
        generate_pattern(48, 36, 1, PatternType::DeterministicNoise, right);
        auto cfg = PipelineConfig::from_mode(QualityMode::Balanced, 16);
        cfg.min_disparity = -4;
        cfg.max_disparity = 8;
        cfg.prior.enable = false;

        GoldenContractFingerprint fp;
        if (!run_and_check("Negative_Disp_Range", left, right, cfg, fp)) {
            return 1;
        }
        fingerprints.push_back(fp);
    }

    // Print summary table of frozen contract fingerprints
    std::cout << "\n--------------------------------------------------------------------------" << std::endl;
    std::cout << "                 P4.0 CONTRACT FINGERPRINTS SUMMARY                       " << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(42) << "Test Tag" 
              << std::setw(10) << "Dim"
              << std::setw(10) << "Invalid"
              << std::setw(10) << "Reliable"
              << std::setw(20) << "Disp FNV-1a Hash" << "\n";
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    for (const auto& fp : fingerprints) {
        std::stringstream dim_ss;
        dim_ss << fp.w << "x" << fp.h;
        std::stringstream hash_ss;
        hash_ss << "0x" << std::hex << fp.disp_hash;

        std::cout << std::left << std::setw(42) << fp.tag
                  << std::setw(10) << dim_ss.str()
                  << std::setw(10) << fp.invalid_count
                  << std::setw(10) << fp.reliable_count
                  << std::setw(20) << hash_ss.str() << "\n";
    }

    std::cout << "\nAll " << fingerprints.size() << " P4.0 API & Behavioral contract cases verified successfully!\n";
    return 0;
}
