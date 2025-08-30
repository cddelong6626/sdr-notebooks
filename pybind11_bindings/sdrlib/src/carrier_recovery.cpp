#include <kfr/base.hpp>
#include <kfr/dft.hpp>
#include <kfr/dsp.hpp>

namespace sdrlib{

float dot(){
    std::vector<float> a_std = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> b_std = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};

    auto a_kfr = kfr::make_univector(a_std);
    auto b_kfr = kfr::make_univector(b_std);

    float res = kfr::sum(a_kfr * b_kfr);

    return res;
}

int add(int a, int b)
{
    return a+b;
}
   
}


// def apply_cfo(signal, pct_offset=0.03, w_offset=None):
//     """Apply carrier frequency offset to signal"""
//     # testing/realistic: 1-5%, aggressive: 10%

//     if w_offset is None:
//         w_offset = pct_offset*(2*np.pi)  # radians/sample
       
//     n = np.arange(len(signal))
//     sig_offset = signal * np.exp(1j*w_offset*n)

//     return sig_offset

// class CostasLoopQPSK {
//     float theta = 0.0f;
//     univector<cpx> buffer;
//     PIDFeedback control;

// public:
//     CostasLoopQPSK(float theta_ = 0, const PIDFeedback<float>& control_, const univector<cpx>& buffer_)
//     : theta(theta), control(control), buffer(buffer) {}

//     univector<cpx> process() {
//         univector<cpx, buffer.size()> symbols_rot;
       
//         // Potentially add vectorized block SIMD rotation using kfr
//         for (size_t i = 0; i < buffer.size(); ++i){
//             symbols_rot[i] = buffer[i] * std::exp(cpx(0, -theta));

//             cpx I = symbols_rot[i].real();
//             cpx Q = symbols_rot[i].imag();

//             // Reference point based on sign of I and Q (+/-1)
//             cpx ref((I >= 0) ? 1.0f : -1.0f, (Q >= 0) ? 1.0f : -1.0f);
           
//             float e = std::arg(symbols_rot[i] / ref)

//             theta += control.update(e);
//         }

//         return symbols_rot
//     }

//     void reset() {
//         theta = 0.0f;
//         control.reset();
//     }
// };
