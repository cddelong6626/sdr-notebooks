
#include <kfr/base.hpp>
#include <complex>

#include "sdrlib/types.hpp"


namespace sdrlib::channel {


void apply_cfo(const sdrlib::cpx* buf_in, sdrlib::cpx* buf_out, size_t n, float w_offset) {
    
    // Convert buffers to KFR univectors
    sdrlib::cvec signal =  kfr::make_univector(buf_in, n);

    // Index expression
    auto idx = kfr::counter(0, 1);

    // Creat a lazy expression for the phase rotation e^(j * n * w_offset)
    sdrlib::cpx j(0.0f, 1.0f);
    auto phase_expr = kfr::cexp(j * idx * w_offset);

    // Multiply signal by phase expression lazily
    auto out_expr = signal * phase_expr;

    // Render the expression and copy the output to buf_out
    kfr::univector_ref<sdrlib::cpx> view_out(buf_out, n);
    view_out = kfr::render(out_expr, n);

}


}