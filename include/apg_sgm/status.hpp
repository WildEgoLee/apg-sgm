#pragma once

#include <string>
#include <utility>

namespace apg {

enum class StatusCode {
    Ok = 0,
    EmptyInput,
    SizeMismatch,
    InvalidDisparityRange,
    InvalidConfig,
    UnsupportedBackend,
    ResourceLimitExceeded,
    AllocationFailure,
    InternalError
};

struct Status {
    StatusCode code = StatusCode::Ok;
    std::string message;

    Status() = default;
    Status(StatusCode c, std::string msg = "") : code(c), message(std::move(msg)) {}

    static Status Ok() {
        return Status(StatusCode::Ok, "");
    }

    static Status success() {
        return Status(StatusCode::Ok, "");
    }

    bool ok() const {
        return code == StatusCode::Ok;
    }

    bool is_ok() const {
        return code == StatusCode::Ok;
    }

    explicit operator bool() const {
        return ok();
    }

    bool operator!() const {
        return !ok();
    }
};

inline Status OkStatus() {
    return Status::Ok();
}

inline const char* status_code_string(StatusCode code) {
    switch (code) {
        case StatusCode::Ok: return "Ok";
        case StatusCode::EmptyInput: return "EmptyInput";
        case StatusCode::SizeMismatch: return "SizeMismatch";
        case StatusCode::InvalidDisparityRange: return "InvalidDisparityRange";
        case StatusCode::InvalidConfig: return "InvalidConfig";
        case StatusCode::UnsupportedBackend: return "UnsupportedBackend";
        case StatusCode::ResourceLimitExceeded: return "ResourceLimitExceeded";
        case StatusCode::AllocationFailure: return "AllocationFailure";
        case StatusCode::InternalError: return "InternalError";
        default: return "Unknown";
    }
}

} // namespace apg
