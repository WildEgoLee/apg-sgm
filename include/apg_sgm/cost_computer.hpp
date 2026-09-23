#pragma once

#include "apg_sgm/buffers.hpp"
#include "apg_sgm/config.hpp"

namespace apg {

class CostComputer {
public:
    void compute_aux(const Image8& left, const Image8& right, const PipelineConfig& cfg,
                     PipelineBuffers& buf) const;

    void compute_volume(const PipelineConfig& cfg, PipelineBuffers& buf) const;

    static uint32_t symmetric_census9x7(const Image8& gray, int x, int y);
    static uint64_t census9x7(const Image8& gray, int x, int y);

    static int popcount32(uint32_t x);
    static int popcount64(uint64_t x);

private:
    void build_gray_and_grad(const Image8& src, Image8& gray, Image8& gx, Image8& gy) const;
    void build_census(const Image8& gray, CensusType type,
                      std::vector<uint32_t>& c32, std::vector<uint64_t>& c64) const;
};

} // namespace apg
