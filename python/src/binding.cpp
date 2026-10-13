#include "apg_sgm/config.hpp"
#include "apg_sgm/execution.hpp"
#include "apg_sgm/image_view.hpp"
#include "apg_sgm/pipeline_stats.hpp"
#include "apg_sgm/result.hpp"
#include "apg_sgm/session.hpp"
#include "apg_sgm/status.hpp"
#include "apg_sgm/types.hpp"
#include "apg_sgm/workspace.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

// Helper to hold buffer data when contiguous copy is required for non-standard striding
struct InputBufferHolder {
    std::vector<uint8_t> fallback_data;
    apg::ImageView8 view;
};

// Extracts an apg::ImageView8 from a Python buffer (NumPy ndarray)
// Implements zero-copy wrapping for row-strided (pitch-padded) C-order arrays.
InputBufferHolder make_image_view(const py::buffer& buf, const std::string& name) {
    py::buffer_info info = buf.request();

    if (info.itemsize != sizeof(uint8_t) ||
        info.format != py::format_descriptor<uint8_t>::format()) {
        throw py::type_error(name + " must have dtype uint8 (got format '" + info.format + "')");
    }

    if (info.ndim != 2 && info.ndim != 3) {
        throw py::value_error(name + " must be 2D (H, W) or 3D (H, W, C), got " + std::to_string(info.ndim) + "D");
    }

    const int h = static_cast<int>(info.shape[0]);
    const int w = static_cast<int>(info.shape[1]);
    const int c = (info.ndim == 3) ? static_cast<int>(info.shape[2]) : 1;

    if (c != 1 && c != 3) {
        throw py::value_error(name + " must have 1 or 3 channels, got " + std::to_string(c));
    }

    if (h <= 0 || w <= 0) {
        throw py::value_error(name + " dimensions must be strictly positive (" + std::to_string(w) + "x" + std::to_string(h) + ")");
    }

    const auto* ptr = static_cast<const uint8_t*>(info.ptr);
    InputBufferHolder holder;

    const py::ssize_t row_stride = info.strides[0];
    const py::ssize_t col_stride = info.strides[1];
    const py::ssize_t ch_stride = (info.ndim == 3) ? info.strides[2] : 1;

    bool zero_copy_candidate = false;
    if (c == 1) {
        if (col_stride == 1 && row_stride >= w) {
            zero_copy_candidate = true;
        } else if (info.ndim == 3 && ch_stride == 1 && col_stride == 1 && row_stride >= w) {
            zero_copy_candidate = true;
        }
    } else if (c == 3) {
        if (ch_stride == 1 && col_stride == 3 && row_stride >= w * 3) {
            zero_copy_candidate = true;
        }
    }

    if (zero_copy_candidate) {
        // Direct zero-copy wrap into ImageView8
        holder.view = apg::ImageView8(ptr, w, h, static_cast<int>(row_stride), c);
    } else {
        // Safe fallback for transposed, negative, or strided views
        holder.fallback_data.resize(static_cast<size_t>(w) * h * c);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (int ch = 0; ch < c; ++ch) {
                    const py::ssize_t offset = (info.ndim == 2)
                        ? (y * row_stride + x * col_stride)
                        : (y * row_stride + x * col_stride + ch * ch_stride);
                    holder.fallback_data[(static_cast<size_t>(y) * w + x) * c + ch] = ptr[offset];
                }
            }
        }
        holder.view = apg::ImageView8(holder.fallback_data.data(), w, h, w * c, c);
    }

    return holder;
}

// Wrapper around StereoResult that binds NumPy views directly to result-owned memory
class PyStereoResult {
public:
    PyStereoResult(apg::StereoResult res, apg::PipelineStats stats)
        : res_(std::move(res)), stats_(std::move(stats)) {}

    int width() const { return res_.width(); }
    int height() const { return res_.height(); }
    bool empty() const { return res_.empty(); }

    const apg::StereoResult& result() const { return res_; }
    const apg::PipelineStats& stats() const { return stats_; }

    std::string repr() const {
        std::ostringstream oss;
        oss << "<apg_sgm.StereoResult " << width() << "x" << height()
            << " valid=" << (empty() ? "False" : "True") << ">";
        return oss.str();
    }

private:
    apg::StereoResult res_;
    apg::PipelineStats stats_;
};

// Python session wrapper managing StereoSession lifecycle and configuration
class PyStereoSession {
public:
    PyStereoSession(const py::object& mode_arg,
                    int threads,
                    int max_disparity,
                    int min_disparity,
                    const py::object& backend_arg,
                    const py::object& memory_policy_arg,
                    size_t max_retained_bytes) {
        apg::QualityMode qm = apg::QualityMode::Balanced;
        if (!mode_arg.is_none()) {
            if (py::isinstance<py::str>(mode_arg)) {
                std::string s = mode_arg.cast<std::string>();
                std::transform(s.begin(), s.end(), s.begin(), ::tolower);
                if (s == "fast") qm = apg::QualityMode::Fast;
                else if (s == "balanced") qm = apg::QualityMode::Balanced;
                else if (s == "high_quality" || s == "hq" || s == "highquality") qm = apg::QualityMode::HighQuality;
                else throw py::value_error("Unrecognized mode '" + s + "'. Valid: 'fast', 'balanced', 'high_quality'");
            } else {
                qm = mode_arg.cast<apg::QualityMode>();
            }
        }

        apg::PipelineConfig cfg = apg::PipelineConfig::from_mode(qm, max_disparity);
        if (min_disparity != 0) {
            cfg.min_disparity = min_disparity;
        }
        if (max_disparity > 0) {
            cfg.max_disparity = max_disparity;
        }
        if (threads > 0) {
            cfg.num_threads = threads;
        }

        if (!backend_arg.is_none()) {
            if (py::isinstance<py::str>(backend_arg)) {
                std::string b = backend_arg.cast<std::string>();
                std::transform(b.begin(), b.end(), b.begin(), ::tolower);
                if (b == "auto") cfg.volume_backend = apg::VolumeBackend::Auto;
                else if (b == "dense") cfg.volume_backend = apg::VolumeBackend::Dense;
                else if (b == "packed") cfg.volume_backend = apg::VolumeBackend::Packed;
                else throw py::value_error("Unrecognized backend '" + b + "'. Valid: 'auto', 'dense', 'packed'");
            } else {
                cfg.volume_backend = backend_arg.cast<apg::VolumeBackend>();
            }
        }

        apg::ExecutionOptions opt;
        opt.memory_policy = apg::MemoryPolicy::ReleasePerFrame; // Default matches P4 C++ engine semantics

        if (!memory_policy_arg.is_none()) {
            if (py::isinstance<py::str>(memory_policy_arg)) {
                std::string p = memory_policy_arg.cast<std::string>();
                std::transform(p.begin(), p.end(), p.begin(), ::tolower);
                if (p == "bounded_reuse" || p == "boundedreuse" || p == "reuse") {
                    opt.memory_policy = apg::MemoryPolicy::BoundedReuse;
                } else if (p == "release_per_frame" || p == "release" || p == "releaseperframe") {
                    opt.memory_policy = apg::MemoryPolicy::ReleasePerFrame;
                } else {
                    throw py::value_error("Unrecognized memory_policy '" + p + "'. Valid: 'bounded_reuse', 'release_per_frame'");
                }
            } else {
                opt.memory_policy = memory_policy_arg.cast<apg::MemoryPolicy>();
            }
        }
        if (max_retained_bytes > 0) {
            opt.max_retained_bytes = max_retained_bytes;
        }

        session_ = std::make_unique<apg::StereoSession>(cfg, opt);
    }

    PyStereoResult process(const py::buffer& left, const py::buffer& right) {
        InputBufferHolder left_holder = make_image_view(left, "left");
        InputBufferHolder right_holder = make_image_view(right, "right");

        if (left_holder.view.width() != right_holder.view.width() ||
            left_holder.view.height() != right_holder.view.height()) {
            throw py::value_error("Left and right images must have identical dimensions (" +
                                  std::to_string(left_holder.view.width()) + "x" +
                                  std::to_string(left_holder.view.height()) + " vs " +
                                  std::to_string(right_holder.view.width()) + "x" +
                                  std::to_string(right_holder.view.height()) + ")");
        }

        if (left_holder.view.channels() != right_holder.view.channels()) {
            throw py::value_error("Left and right images must have matching channel counts (" +
                                  std::to_string(left_holder.view.channels()) + " vs " +
                                  std::to_string(right_holder.view.channels()) + ")");
        }

        apg::StereoResult res;
        apg::PipelineStats stats;
        apg::Status s;

        // Concurrency semantics: calls on the same StereoSession instance are serialized via mutex_,
        // while calls across independent StereoSession instances execute concurrently with GIL released.
        // Lock ordering: release GIL first, acquire mutex_ second; mutex_ is released before re-acquiring GIL.
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            s = session_->process_status(left_holder.view, right_holder.view, res, &stats);
        }

        if (!s.ok()) {
            throw std::runtime_error("Stereo processing failed [" +
                                     std::to_string(static_cast<int>(s.code)) + "]: " + s.message);
        }

        return PyStereoResult(std::move(res), stats);
    }

    void reset() {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mutex_);
        session_->reset();
    }

    size_t frame_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_->frame_count();
    }
    double total_time_ms() {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_->total_time_ms();
    }
    double average_time_ms() {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_->average_time_ms();
    }
    size_t allocated_bytes() {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_->allocated_bytes();
    }
    std::string last_error() {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_->last_error();
    }

    std::string repr() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream oss;
        oss << "<apg_sgm.StereoSession frames=" << session_->frame_count()
            << " avg_time_ms=" << session_->average_time_ms()
            << " allocated_bytes=" << session_->allocated_bytes() << ">";
        return oss.str();
    }

private:
    std::unique_ptr<apg::StereoSession> session_;
    std::mutex mutex_;
};

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "APG-SGM Stereo Engine Python bindings (Zero-copy StereoSession API)";

    // QualityMode Enum
    py::enum_<apg::QualityMode>(m, "QualityMode", "Algorithm quality presets")
        .value("Fast", apg::QualityMode::Fast)
        .value("Balanced", apg::QualityMode::Balanced)
        .value("HighQuality", apg::QualityMode::HighQuality)
        .export_values();

    // VolumeBackend Enum
    py::enum_<apg::VolumeBackend>(m, "VolumeBackend", "Cost volume memory layout backend")
        .value("Auto", apg::VolumeBackend::Auto)
        .value("Dense", apg::VolumeBackend::Dense)
        .value("Packed", apg::VolumeBackend::Packed)
        .export_values();

    // MemoryPolicy Enum
    py::enum_<apg::MemoryPolicy>(m, "MemoryPolicy", "Workspace memory retention policy")
        .value("ReleasePerFrame", apg::MemoryPolicy::ReleasePerFrame)
        .value("BoundedReuse", apg::MemoryPolicy::BoundedReuse)
        .export_values();

    // StatusCode Enum
    py::enum_<apg::StatusCode>(m, "StatusCode", "Execution status codes")
        .value("Ok", apg::StatusCode::Ok)
        .value("EmptyInput", apg::StatusCode::EmptyInput)
        .value("SizeMismatch", apg::StatusCode::SizeMismatch)
        .value("InvalidDisparityRange", apg::StatusCode::InvalidDisparityRange)
        .value("InvalidConfig", apg::StatusCode::InvalidConfig)
        .value("UnsupportedBackend", apg::StatusCode::UnsupportedBackend)
        .value("ResourceLimitExceeded", apg::StatusCode::ResourceLimitExceeded)
        .value("AllocationFailure", apg::StatusCode::AllocationFailure)
        .value("InternalError", apg::StatusCode::InternalError)
        .export_values();

    // StageTiming Struct
    py::class_<apg::StageTiming>(m, "StageTiming", "Execution timings per stage in milliseconds")
        .def_readonly("aux_ms", &apg::StageTiming::aux_ms)
        .def_readonly("prior_ms", &apg::StageTiming::prior_ms)
        .def_readonly("cost_ms", &apg::StageTiming::cost_ms)
        .def_readonly("right_wta_ms", &apg::StageTiming::right_wta_ms)
        .def_readonly("cross_ms", &apg::StageTiming::cross_ms)
        .def_readonly("sgm_ms", &apg::StageTiming::sgm_ms)
        .def_readonly("wta_ms", &apg::StageTiming::wta_ms)
        .def_readonly("confidence_ms", &apg::StageTiming::confidence_ms)
        .def_readonly("refine_ms", &apg::StageTiming::refine_ms)
        .def_readonly("post_ms", &apg::StageTiming::post_ms)
        .def_readonly("total_ms", &apg::StageTiming::total_ms)
        .def("to_dict", [](const apg::StageTiming& t) {
            py::dict d;
            d["aux_ms"] = t.aux_ms;
            d["prior_ms"] = t.prior_ms;
            d["cost_ms"] = t.cost_ms;
            d["right_wta_ms"] = t.right_wta_ms;
            d["cross_ms"] = t.cross_ms;
            d["sgm_ms"] = t.sgm_ms;
            d["wta_ms"] = t.wta_ms;
            d["confidence_ms"] = t.confidence_ms;
            d["refine_ms"] = t.refine_ms;
            d["post_ms"] = t.post_ms;
            d["total_ms"] = t.total_ms;
            return d;
        })
        .def("__repr__", [](const apg::StageTiming& t) {
            std::ostringstream oss;
            oss << "<StageTiming total=" << t.total_ms << "ms (sgm=" << t.sgm_ms
                << "ms, cost=" << t.cost_ms << "ms)>";
            return oss.str();
        });

    // PipelineStats Struct
    py::class_<apg::PipelineStats>(m, "PipelineStats", "Comprehensive execution diagnostics and metrics")
        .def_readonly("timing", &apg::PipelineStats::timing)
        .def_readonly("peak_workspace_bytes", &apg::PipelineStats::peak_workspace_bytes)
        .def_readonly("retained_workspace_bytes", &apg::PipelineStats::retained_workspace_bytes)
        .def_readonly("cost_bytes", &apg::PipelineStats::cost_bytes)
        .def_readonly("estimated_peak_bytes", &apg::PipelineStats::estimated_peak_bytes)
        .def_readonly("valid_disparity_ratio", &apg::PipelineStats::valid_disparity_ratio)
        .def_readonly("reliable_ratio", &apg::PipelineStats::reliable_ratio)
        .def_readonly("unreliable_ratio", &apg::PipelineStats::unreliable_ratio)
        .def_readonly("refine_changed_ratio", &apg::PipelineStats::refine_changed_ratio)
        .def_readonly("lr_fail_ratio", &apg::PipelineStats::lr_fail_ratio)
        .def_readonly("effective_threads", &apg::PipelineStats::effective_threads)
        .def_readonly("requested_threads", &apg::PipelineStats::requested_threads)
        .def_readonly("resolved_backend", &apg::PipelineStats::resolved_backend)
        .def_readonly("requested_backend", &apg::PipelineStats::requested_backend)
        .def_readonly("memory_policy", &apg::PipelineStats::memory_policy)
        .def("to_dict", [](const apg::PipelineStats& s) {
            py::dict d;
            d["timing"] = py::cast(s.timing).attr("to_dict")();
            d["peak_workspace_bytes"] = s.peak_workspace_bytes;
            d["retained_workspace_bytes"] = s.retained_workspace_bytes;
            d["valid_disparity_ratio"] = s.valid_disparity_ratio;
            d["reliable_ratio"] = s.reliable_ratio;
            d["unreliable_ratio"] = s.unreliable_ratio;
            d["effective_threads"] = s.effective_threads;
            d["resolved_backend"] = s.resolved_backend;
            d["memory_policy"] = s.memory_policy;
            return d;
        })
        .def("__repr__", [](const apg::PipelineStats& s) {
            std::ostringstream oss;
            oss << "<PipelineStats total=" << s.timing.total_ms << "ms"
                << " valid_ratio=" << s.valid_disparity_ratio
                << " peak_ws_mb=" << (s.peak_workspace_bytes / (1024.0 * 1024.0)) << ">";
            return oss.str();
        });

    // StereoResult Class
    py::class_<PyStereoResult>(m, "StereoResult", "Result output holding disparity, confidence, and reliable mask")
        .def_property_readonly("width", &PyStereoResult::width)
        .def_property_readonly("height", &PyStereoResult::height)
        .def_property_readonly("empty", &PyStereoResult::empty)
        .def_property_readonly("stats", &PyStereoResult::stats)
        .def_property_readonly("disparity", [](py::object self) {
            const auto& r = self.cast<PyStereoResult&>();
            if (r.result().disparity.empty()) return py::array_t<float>();
            const int h = r.height();
            const int w = r.width();
            return py::array_t<float>(
                {h, w},
                {static_cast<py::ssize_t>(w * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
                r.result().disparity.data(),
                self // base: keeps PyStereoResult alive as long as NumPy view is alive!
            );
        })
        .def_property_readonly("confidence", [](py::object self) {
            const auto& r = self.cast<PyStereoResult&>();
            if (r.result().confidence.empty()) return py::array_t<float>();
            const int h = r.height();
            const int w = r.width();
            return py::array_t<float>(
                {h, w},
                {static_cast<py::ssize_t>(w * sizeof(float)), static_cast<py::ssize_t>(sizeof(float))},
                r.result().confidence.data(),
                self
            );
        })
        .def_property_readonly("reliable_mask", [](py::object self) {
            const auto& r = self.cast<PyStereoResult&>();
            if (r.result().reliable_mask.empty()) return py::array_t<uint8_t>();
            const int h = r.height();
            const int w = r.width();
            return py::array_t<uint8_t>(
                {h, w},
                {static_cast<py::ssize_t>(w * sizeof(uint8_t)), static_cast<py::ssize_t>(sizeof(uint8_t))},
                r.result().reliable_mask.data(),
                self
            );
        })
        .def_property_readonly("invalid_reason", [](py::object self) {
            const auto& r = self.cast<PyStereoResult&>();
            if (r.result().invalid_reason.empty()) return py::array_t<uint8_t>();
            const int h = r.height();
            const int w = r.width();
            return py::array_t<uint8_t>(
                {h, w},
                {static_cast<py::ssize_t>(w * sizeof(uint8_t)), static_cast<py::ssize_t>(sizeof(uint8_t))},
                r.result().invalid_reason.data(),
                self
            );
        })
        .def("__repr__", &PyStereoResult::repr);

    // StereoSession Class
    py::class_<PyStereoSession>(m, "StereoSession", "Stateful streaming stereo matcher session")
        .def(py::init<py::object, int, int, int, py::object, py::object, size_t>(),
             py::arg("mode") = "balanced",
             py::arg("threads") = 0,
             py::arg("max_disparity") = 128,
             py::arg("min_disparity") = 0,
             py::arg("backend") = "auto",
             py::arg("memory_policy") = "release_per_frame",
             py::arg("max_retained_bytes") = 0,
             "Initialize StereoSession with algorithm mode, disparity bounds, and execution options.")
        .def("process", &PyStereoSession::process, py::arg("left"), py::arg("right"),
             "Process a pair of left and right images (uint8 ndarrays). Returns StereoResult.")
        .def("compute", &PyStereoSession::process, py::arg("left"), py::arg("right"),
             "Alias for process().")
        .def("reset", &PyStereoSession::reset, "Release retained workspace storage and reset frame counters.")
        .def_property_readonly("frame_count", &PyStereoSession::frame_count)
        .def_property_readonly("total_time_ms", &PyStereoSession::total_time_ms)
        .def_property_readonly("average_time_ms", &PyStereoSession::average_time_ms)
        .def_property_readonly("allocated_bytes", &PyStereoSession::allocated_bytes)
        .def_property_readonly("last_error", &PyStereoSession::last_error)
        .def("__repr__", &PyStereoSession::repr);
}
