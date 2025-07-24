


class CostasLoopQPSK {
    float theta = 0.0f;
    univector<cpx> buffer;
    PIDFeedback control;

public:
    CostasLoopQPSK(float theta_ = 0, const PIDFeedback<float>& control_, const univector<cpx>& buffer_)
    : theta(theta), control(control), buffer(buffer) {}

    univector<cpx> process() {
        univector<cpx, buffer.size()> symbols_rot;
       
        // Potentially add vectorized block SIMD rotation using kfr
        for (size_t i = 0; i < buffer.size(); ++i){
            symbols_rot[i] = buffer[i] * std::exp(cpx(0, -theta));

            cpx I = symbols_rot[i].real();
            cpx Q = symbols_rot[i].imag();

            // Reference point based on sign of I and Q (+/-1)
            cpx ref((I >= 0) ? 1.0f : -1.0f, (Q >= 0) ? 1.0f : -1.0f);
           
            float e = std::arg(symbols_rot[i] / ref)

            theta += control.update(e);
        }

        return symbols_rot
    }

    void reset() {
        theta = 0.0f;
        control.reset();
    }
};
