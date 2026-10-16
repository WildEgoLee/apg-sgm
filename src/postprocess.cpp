#include "apg_sgm/postprocess.hpp"
#include "inpaint_internal.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace apg {

void PostProcessor::left_right_check(const PipelineConfig& cfg, PipelineBuffers& buf) const {
    if (!cfg.post.lr_check || buf.disparity_right.empty()) return;
    const int w = buf.disparity.width();
    const int h = buf.disparity.height();
    const float tol = static_cast<float>(cfg.post.lr_max_diff) + 0.5f;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
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
    const InpaintMethod method = ExperimentalInpaintScope::current_method();
    const bool telem_enabled = ExperimentalInpaintScope::is_telemetry_enabled();
    const int scope_gap = ExperimentalInpaintScope::current_max_gap();
    const int max_gap = (method == InpaintMethod::Legacy && scope_gap == 32) ? cfg.post.max_fill_gap : scope_gap;
    const bool is_fast_legacy = (method == InpaintMethod::Legacy && !telem_enabled && max_gap == cfg.post.max_fill_gap);

    Image32f out = buf.disparity;
    InpaintTelemetry* caller_telem = telem_enabled ? &ExperimentalInpaintScope::current_telemetry() : nullptr;

#if defined(_OPENMP)
    #pragma omp parallel
#endif
    {
        InpaintTelemetry local_telem;

#if defined(_OPENMP)
        #pragma omp for schedule(dynamic, 1)
#endif
        for (int y = 0; y < h; ++y) {
            if (telem_enabled) {
                // Record invalid run lengths along scanline y
                bool in_run = false;
                int cur_run_len = 0;
                for (int x = 0; x < w; ++x) {
                    if (buf.disparity.at(x, y) < 0.f) {
                        if (!in_run) {
                            in_run = true;
                            cur_run_len = 1;
                        } else {
                            cur_run_len++;
                        }
                    } else if (in_run) {
                        local_telem.record_run(cur_run_len, max_gap);
                        in_run = false;
                        cur_run_len = 0;
                    }
                }
                if (in_run) {
                    local_telem.record_run(cur_run_len, max_gap);
                }
            }

            for (int x = 0; x < w; ++x) {
                if (buf.disparity.at(x, y) >= 0.f) continue;
                const auto reason = static_cast<InvalidReason>(buf.invalid_reason.at(x, y));

                int xl = x - 1;
                while (xl >= 0 && xl >= x - max_gap && buf.disparity.at(xl, y) < 0.f) --xl;
                int xr = x + 1;
                while (xr < w && xr <= x + max_gap && buf.disparity.at(xr, y) < 0.f) ++xr;

                const bool hit_bound = (xl < 0 || (x - xl) > max_gap || xr >= w || (xr - x) > max_gap);
                const float dl = (xl >= 0 && buf.disparity.at(xl, y) >= 0.f) ? buf.disparity.at(xl, y) : -1.f;
                const float dr = (xr < w && buf.disparity.at(xr, y) >= 0.f) ? buf.disparity.at(xr, y) : -1.f;

                if (telem_enabled) {
                    local_telem.total_invalid_pixels++;
                    if (reason == InvalidReason::Occlusion) local_telem.classified_occ_pixels++;
                    else if (reason == InvalidReason::Mismatch) local_telem.classified_mismatch_pixels++;
                    if (hit_bound) local_telem.hit_max_gap_pixels++;
                }

                if (is_fast_legacy || method == InpaintMethod::Legacy) {
                    if (reason == InvalidReason::Occlusion) {
                        if (dl >= 0.f && dr >= 0.f) {
                            out.at(x, y) = std::min(dl, dr);
                            if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::OccBothMin)]++;
                        } else if (dl >= 0.f) {
                            out.at(x, y) = dl;
                            if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::OccLeftOnly)]++;
                        } else if (dr >= 0.f) {
                            out.at(x, y) = dr;
                            if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::OccRightOnly)]++;
                        } else {
                            if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::RemainedInvalid)]++;
                        }
                        continue;
                    }

                    // Mismatch
                    const uint8_t c0 = buf.left_gray.at(x, y);
                    float best_d = -1.f;
                    int best_abs = 999;
                    bool best_is_vert = false;

                    auto consider = [&](int xx, int yy, bool is_vert) {
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) return;
                        const float d = buf.disparity.at(xx, yy);
                        if (d < 0.f) return;
                        const int ad = std::abs(static_cast<int>(buf.left_gray.at(xx, yy)) - static_cast<int>(c0));
                        if (ad < best_abs) {
                            best_abs = ad;
                            best_d = d;
                            best_is_vert = is_vert;
                        }
                    };
                    consider(xl, y, false);
                    consider(xr, y, false);
                    if (y > 0) consider(x, y - 1, true);
                    if (y + 1 < h) consider(x, y + 1, true);

                    if (best_d >= 0.f && best_abs <= 18) {
                        out.at(x, y) = best_d;
                        if (telem_enabled) {
                            local_telem.route_counts[static_cast<size_t>(best_is_vert ? FillRoute::MismatchColorVert : FillRoute::MismatchColorHoriz)]++;
                        }
                    } else if (!buf.d_prior.empty() && buf.d_prior.at(x, y) >= 0.f) {
                        out.at(x, y) = buf.d_prior.at(x, y);
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchPrior)]++;
                    } else if (dl >= 0.f && dr >= 0.f) {
                        const float t = static_cast<float>(x - xl) / static_cast<float>(xr - xl);
                        out.at(x, y) = dl * (1.f - t) + dr * t;
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchLerp)]++;
                    } else if (dl >= 0.f) {
                        out.at(x, y) = dl;
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchLeftOnly)]++;
                    } else if (dr >= 0.f) {
                        out.at(x, y) = dr;
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchRightOnly)]++;
                    } else {
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::RemainedInvalid)]++;
                    }
                    continue;
                }

                // Candidate C: Shadow-aware mismatch fill
                const bool is_occ = (reason == InvalidReason::Occlusion);
                const bool is_shadow_step = (method == InpaintMethod::ShadowReclass &&
                                             reason == InvalidReason::Mismatch &&
                                             dl >= 0.f && dr >= 0.f &&
                                             std::abs(dr - dl) >= 4.0f);

                if (is_occ || is_shadow_step) {
                    if (telem_enabled && is_shadow_step) {
                        local_telem.reclassified_shadow_pixels++;
                    }
                    if (dl >= 0.f && dr >= 0.f) {
                        out.at(x, y) = std::min(dl, dr);
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::OccBothMin)]++;
                    } else if (dl >= 0.f) {
                        out.at(x, y) = dl;
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::OccLeftOnly)]++;
                    } else if (dr >= 0.f) {
                        out.at(x, y) = dr;
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::OccRightOnly)]++;
                    } else {
                        if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::RemainedInvalid)]++;
                    }
                    continue;
                }

                // Smooth Mismatch (not a shadow step)
                const uint8_t c0 = buf.left_gray.at(x, y);
                float best_d = -1.f;
                int best_abs = 999;
                bool best_is_vert = false;

                auto consider = [&](int xx, int yy, bool is_vert) {
                    if (xx < 0 || yy < 0 || xx >= w || yy >= h) return;
                    const float d = buf.disparity.at(xx, yy);
                    if (d < 0.f) return;
                    const int ad = std::abs(static_cast<int>(buf.left_gray.at(xx, yy)) - static_cast<int>(c0));
                    if (ad < best_abs) {
                        best_abs = ad;
                        best_d = d;
                        best_is_vert = is_vert;
                    }
                };
                consider(xl, y, false);
                consider(xr, y, false);
                if (y > 0) consider(x, y - 1, true);
                if (y + 1 < h) consider(x, y + 1, true);

                if (best_d >= 0.f && best_abs <= 18) {
                    out.at(x, y) = best_d;
                    if (telem_enabled) {
                        local_telem.route_counts[static_cast<size_t>(best_is_vert ? FillRoute::MismatchColorVert : FillRoute::MismatchColorHoriz)]++;
                    }
                } else if (!buf.d_prior.empty() && buf.d_prior.at(x, y) >= 0.f) {
                    out.at(x, y) = buf.d_prior.at(x, y);
                    if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchPrior)]++;
                } else if (dl >= 0.f && dr >= 0.f) {
                    const float t = static_cast<float>(x - xl) / static_cast<float>(xr - xl);
                    out.at(x, y) = dl * (1.f - t) + dr * t;
                    if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchLerp)]++;
                } else if (dl >= 0.f) {
                    out.at(x, y) = dl;
                    if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchLeftOnly)]++;
                } else if (dr >= 0.f) {
                    out.at(x, y) = dr;
                    if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::MismatchRightOnly)]++;
                } else {
                    if (telem_enabled) local_telem.route_counts[static_cast<size_t>(FillRoute::RemainedInvalid)]++;
                }
            }
        }

        if (caller_telem) {
#if defined(_OPENMP)
            #pragma omp critical(inpaint_telem_merge)
#endif
            {
                caller_telem->merge(local_telem);
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
    const size_t max_win = static_cast<size_t>((2 * r + 1) * (2 * r + 1));

#if defined(_OPENMP)
    #pragma omp parallel
    {
        std::vector<float> win;
        win.reserve(max_win);
        #pragma omp for schedule(dynamic, 1)
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
    }
#else
    std::vector<float> win;
    win.reserve(max_win);
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
#endif
    buf.disparity = std::move(out);
}

} // namespace apg
