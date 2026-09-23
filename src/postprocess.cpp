#include "apg_sgm/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace apg {

void PostProcessor::left_right_check(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    if (!cfg.post.lr_check || buf.disparity_right.empty()) return;
    const int w = buf.disparity.width();
    const int h = buf.disparity.height();
    const float tol = static_cast<float>(cfg.post.lr_max_diff) + 0.5f;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float d = buf.disparity.at(x, y);
            if (d < 0.f) continue;
            const int xr = static_cast<int>(std::round(static_cast<float>(x) - d));
            if (xr < 0 || xr >= w) {
                buf.disparity.at(x, y) = -1.f;
                buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Occlusion);
                continue;
            }
            const float dr = buf.disparity_right.at(xr, y);
            if (dr < 0.f || std::abs(d - dr) > tol) {
                buf.disparity.at(x, y) = -1.f;
                if (dr > d + tol) {
                    buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Occlusion);
                } else {
                    buf.invalid_reason.at(x, y) = static_cast<uint8_t>(InvalidReason::Mismatch);
                }
            }
        }
    }
}

void PostProcessor::fill_holes(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int w = buf.disparity.width();
    const int h = buf.disparity.height();
    const int max_gap = cfg.post.max_fill_gap;

    Image32f out = buf.disparity;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (buf.disparity.at(x, y) >= 0.f) continue;
            const auto reason = static_cast<InvalidReason>(buf.invalid_reason.at(x, y));

            int xl = x - 1;
            while (xl >= 0 && xl >= x - max_gap && buf.disparity.at(xl, y) < 0.f) --xl;
            int xr = x + 1;
            while (xr < w && xr <= x + max_gap && buf.disparity.at(xr, y) < 0.f) ++xr;

            float dl = (xl >= 0 && buf.disparity.at(xl, y) >= 0.f) ? buf.disparity.at(xl, y) : -1.f;
            float dr = (xr < w && buf.disparity.at(xr, y) >= 0.f) ? buf.disparity.at(xr, y) : -1.f;

            if (reason == InvalidReason::Occlusion) {
                if (dl >= 0.f && dr >= 0.f) out.at(x, y) = std::min(dl, dr);
                else if (dl >= 0.f) out.at(x, y) = dl;
                else if (dr >= 0.f) out.at(x, y) = dr;
                continue;
            }

            const uint8_t c0 = buf.left_gray.at(x, y);
            float best_d = -1.f;
            int best_abs = 999;
            auto consider = [&](int xx, int yy) {
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) return;
                const float d = buf.disparity.at(xx, yy);
                if (d < 0.f) return;
                const int ad = std::abs(static_cast<int>(buf.left_gray.at(xx, yy)) - static_cast<int>(c0));
                if (ad < best_abs) {
                    best_abs = ad;
                    best_d = d;
                }
            };
            consider(xl, y);
            consider(xr, y);
            if (y > 0) consider(x, y - 1);
            if (y + 1 < h) consider(x, y + 1);
            if (best_d >= 0.f && best_abs <= 18) {
                out.at(x, y) = best_d;
            } else if (!buf.d_prior.empty() && buf.d_prior.at(x, y) >= 0.f) {
                out.at(x, y) = buf.d_prior.at(x, y);
            } else if (dl >= 0.f && dr >= 0.f) {
                const float t = static_cast<float>(x - xl) / static_cast<float>(xr - xl);
                out.at(x, y) = dl * (1.f - t) + dr * t;
            } else if (dl >= 0.f) {
                out.at(x, y) = dl;
            } else if (dr >= 0.f) {
                out.at(x, y) = dr;
            }
        }
    }
    buf.disparity = std::move(out);
}

void PostProcessor::median(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    const int r = cfg.post.median_radius;
    if (r <= 0) return;
    const int w = buf.disparity.width();
    const int h = buf.disparity.height();
    Image32f out = buf.disparity;
    std::vector<float> win;
    win.reserve(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            win.clear();
            for (int oy = -r; oy <= r; ++oy) {
                for (int ox = -r; ox <= r; ++ox) {
                    const int xx = x + ox, yy = y + oy;
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
                    const float d = buf.disparity.at(xx, yy);
                    if (d >= 0.f) win.push_back(d);
                }
            }
            if (win.empty()) continue;
            const size_t mid = win.size() / 2;
            std::nth_element(win.begin(), win.begin() + static_cast<std::ptrdiff_t>(mid), win.end());
            out.at(x, y) = win[mid];
        }
    }
    buf.disparity = std::move(out);
}

} // namespace apg
