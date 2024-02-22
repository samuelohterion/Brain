#ifndef BRAIN_HPP
#define BRAIN_HPP

#include <math.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <deque>
// #include <numeric>
#include "../AlgebraWithSTL/algebra.hpp"
// #include "algebra.hpp" // for buildBrain.m

template <typename T = int>
class QueueSum : public std::deque<T> {

    private:
    
        T
        __sum;

    public:

        QueueSum(std::size_t const &p_size) :
        std::deque<T>(p_size),
        __sum(0) {
        }

        QueueSum<T> &
        add(T const &p_value) {
            __sum -= this->back();
            this->pop_back();
            this->push_front(p_value);
            __sum += p_value;
            return *this;
        }

        T
        sum() const {
            return __sum;
        }

        T
        mean() const {
            return __sum / this->size();
        }
};

class Brain {

    private:

        class Sig {

            public:

                double
                mn, mx, dst;

            public:

                Sig(double const &p_min = 0, double const &p_max = 1.) :
                mn(p_min), mx(p_max), dst(mx - mn) {
            }

            double
            operator()(double const &p_net) const {
                return mn + dst / (1. + exp(-p_net));
            }
        };

        class DSig :
        public Sig {
        
            public:
        
                double
                mn, mx, dstRec, flat_spot_elimination_offset;

            public:
                DSig(double const &p_min = 0., double const &p_max = 1., double const & p_flat_spot_elimination_offset = .001) :
                mn(p_min), mx(p_max), dstRec(1 / (mx - mn)) {
            }

            double
            operator()(double const &p_act) const {
                // return (mx - p_act) * (p_act - mn) / (mx - mn);
                return flat_spot_elimination_offset + (mx - p_act) * (p_act - mn) * dstRec;
            }
        };

    public:
    
        std::vector<std::size_t>
        layer_sizes;

        Sig
        act;

        DSig
        dact;

        double
        eta0, eta, eta_halftime, delta_eta, weights_min, weights_max, adam_beta1, adam_beta2;

        std::size_t
        batch_size, batch_count,
        step,
        storing_loop,
        storing_period;

        std::vector<std::vector<double>>
        o, // utput
        d; // elta

        std::vector<std::vector<std::vector<double>>>
        w, // eights
        d_w, s_w,
        adam_m, adam_v;

        std::vector<std::vector<std::vector<std::vector<double>>>>
        m; // emory

    public:

        Brain(
            std::initializer_list<std::size_t> const &p_layer_sizes,
            double const &p_eta = .001,
            double const &p_eta_halftime = 1e9,
            double const &p_delta_eta = 1.,
            double const &p_activation_min = 0.,
            double const &p_activation_max = 1.,
            double const &p_weights_min = -1.,
            double const &p_weights_max = 1.,
            std::size_t const &p_seed = static_cast<std::size_t>(time(nullptr)),
            std::size_t const &p_storing_period = 0,
            std::size_t const &p_batch_size = 0) :
        layer_sizes(p_layer_sizes.begin(), p_layer_sizes.end()),
        act(p_activation_min, p_activation_max), dact(p_activation_min, p_activation_max),
        eta0(p_eta), eta(eta0), eta_halftime(p_eta_halftime), delta_eta(p_delta_eta),
        weights_min(p_weights_min), weights_max(p_weights_max),
        batch_size(p_batch_size), batch_count(0),
        step(0),
        storing_loop(0), storing_period(p_storing_period) {
            configure(p_weights_min, p_weights_max, p_seed);
        }

        Brain(
            std::vector<std::size_t> const &p_layer_sizes,
            double const &p_eta = .001,
            double const &p_eta_halftime = 1e9,
            double const &p_delta_eta = 1.,
            double const &p_activation_min = 0.,
            double const &p_activation_max = 1.,
            double const &p_weights_min = -1.,
            double const &p_weights_max = 1.,
            std::size_t const &p_seed = static_cast<std::size_t>(time(nullptr)),
            std::size_t const &p_storing_period = 0,
            std::size_t const &p_batch_size = 0) :
        layer_sizes(p_layer_sizes.begin(), p_layer_sizes.end()),
        act(p_activation_min, p_activation_max), dact(p_activation_min, p_activation_max),
        eta0(p_eta), eta(eta0), eta_halftime(p_eta_halftime), delta_eta(p_delta_eta),
        weights_min(p_weights_min), weights_max(p_weights_max),
        batch_size(p_batch_size), batch_count(0),
        step(0),
        storing_loop(0), storing_period(p_storing_period) {
            configure(p_weights_min, p_weights_max, p_seed);
        }

        Brain(std::string const &p_filename) {
            loadMe(p_filename);
        }

        static std::vector<double>
        digitize(double const &p_x, double const &p_x0, double const &p_x1, std::size_t p_digits) {
            double x = (p_x - p_x0) / (p_x1 - p_x0);
            x =
                x < 0
                    ? 0
                    : 1 < x
                        ? 1
                        : x;

            std::vector<double>
            bit(p_digits);

            while (0 < p_digits) {
                --p_digits;
                if (x < .5) {
                    bit[p_digits] = 0;
                }
                else {
                    bit[p_digits] = 1;
                    x -= .5;
                }
                x *= 2;
            }
            return bit;
        }

        static double
        analogize(std::vector<double> p_bits, double const &p_x0, double const &p_x1) {
            double
            sum = 0.,
            part = .5;
            std::size_t
            i = p_bits.size();
            while (0 < i) {
                sum += .5 < p_bits[--i] ? part : 0.;
                part *= .5;
            }
            return p_x0 + (p_x1 - p_x0) * sum;
        }

        Brain &
        configure(double const &p_weights_min = 0., double const &p_weights_max = 1., std::size_t const &p_seed = std::time(nullptr)) {
            weights_min = p_weights_min;
            weights_max = p_weights_max;
            for (std::size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                o.push_back(std::vector<double>(layer_sizes[i] + 1, 1.));
            }
            o.push_back(std::vector<double>(layer_sizes[layer_sizes.size() - 1], 1.));
            for (std::size_t i = 1; i < layer_sizes.size(); ++i) {
                d.push_back(std::vector<double>(layer_sizes[i]));
            }
            for (std::size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                std::size_t
                realNumberOfNeuoronsInLayer = layer_sizes[i + 1],
                realNumberOfNeuoronsInPrevLayer = layer_sizes[i] + 1;
                w.push_back(std::vector<std::vector<double>>(realNumberOfNeuoronsInLayer, std::vector<double>(realNumberOfNeuoronsInPrevLayer)));
                d_w.push_back(std::vector<std::vector<double>>(realNumberOfNeuoronsInLayer, std::vector<double>(realNumberOfNeuoronsInPrevLayer)));
                s_w.push_back(std::vector<std::vector<double>>(realNumberOfNeuoronsInLayer, std::vector<double>(realNumberOfNeuoronsInPrevLayer)));
                adam_m.push_back(std::vector<std::vector<double>>(realNumberOfNeuoronsInLayer, std::vector<double>(realNumberOfNeuoronsInPrevLayer)));
                adam_v.push_back(std::vector<std::vector<double>>(realNumberOfNeuoronsInLayer, std::vector<double>(realNumberOfNeuoronsInPrevLayer)));
            }
            randomizeWeights(p_seed);
            adam_beta1 = .9;
            adam_beta2 = .999;
            return *this;
        }

        double
        error(std::vector<double> const &p_teacher) const {
            std::size_t
            lyr = layer_sizes.size() - 1,
            to  = layer_sizes[lyr];
            double
            r = 0,
            tmp;
            for (std::size_t i = 0; i < to; ++i) {
                tmp = o[lyr][i] - p_teacher[i];
                r += tmp * tmp;
            }
            return sqrt(r);
        }

        double
        rms() const {
            std::size_t
            lyr = d.size() - 1,
            to = d[lyr].size();
            double
            r = 0;
            for (std::size_t i = 0; i < to; ++i) {
                r += d[lyr][i] * d[lyr][i];
            }
            return sqrt(r / to);
        }

        struct
        ERROR {
            std::size_t
            minId, maxId;
            double
            minVal, maxVal, sum;
            ERROR() :
            minId(0), maxId(0), sum(0.) {
            }
            std::string str() const {
                std::stringstream ss;
                ss << "Error summary:" << std::endl
                << "  min id   : " << this->minId << std::endl
                << "  min val  : " << this->minVal << std::endl
                << "  max id   : " << this->maxId << std::endl
                << "  max val  : " << this->maxVal << std::endl
                << "  err^2 tot: " << this->sum << std::endl
                << "  err tot: " << sqrt(this->sum) << std::endl;
                return ss.str();
            }
        };

        ERROR
        errorTotal(std::vector<std::vector<double>> const &p_patterns, std::vector<std::vector<double>> const &p_teachers) {
            ERROR
            err;
            std::size_t
            i = 0;
            remember(p_patterns[i]);
            double
            errVal = error(p_teachers[i]);
            err.sum = errVal;
            err.minId = err.maxId = i;
            err.minVal = err.maxVal = errVal;
            while (++i < p_teachers.size()) {
                remember(p_patterns[i]);
                errVal = error(p_teachers[i]);
                err.sum += errVal;
                if (errVal < err.minVal) {
                    err.minId = i;
                    err.minVal = errVal;
                }
                if (err.maxVal < errVal) {
                    err.maxId = i;
                    err.maxVal = errVal;
                }
            }
            return err;
        }

        std::vector<std::vector<std::vector<std::vector<double>>>> &
        history() {
            return m;
        }

        bool
        loadHistory(std::string const &p_filename) {
            return alg::load(p_filename, m);
        }

        bool
        loadWeights(std::string const &p_filename) {
            return alg::load(p_filename, w);
        }

        std::vector<std::vector<std::vector<double>>> const &history(std::size_t const &p_id) const {
            return m[p_id];
        }

        std::vector<std::vector<double>> const &history(std::size_t const &p_id, std::size_t const &p_layer) const {
            return m[p_id][p_layer];
        }

        std::vector<double> const &
        history(std::size_t const &p_id, std::size_t const &p_layer, std::size_t const &p_row) const {
            return m[p_id][p_layer][p_row];
        }

        std::vector<double> const &input() const {
            return o[0];
        }

        double const &
        input(std::size_t const &p_index) const {
            return o[0][p_index];
        }

        void
        norm() {
            double
            maxOut = 0.;
            for (std::size_t l = 0; l < w.size(); ++l) {
                for (std::size_t r = 0; r < w[l].size(); ++r) {
                    for (std::size_t c = 0; c < w[l][r].size(); ++c) {
                        if (maxOut < abs(w[l][r][c])) {
                            maxOut = abs(w[l][r][c]);
                        }
                    }
                }
            }
            if (maxOut < 1e-6) {
                return;
            }
            for (std::size_t l = 0; l < w.size(); ++l) {
                for (std::size_t r = 0; r < w[l].size(); ++r) {
                    for (std::size_t c = 0; c < w[l][r].size(); ++c) {
                        if (maxOut < abs(w[l][r][c])) {
                            w[l][r][c] /= maxOut;
                        }
                    }
                }
            }
        }

        std::vector<double> const &output() const {
            return o[o.size() - 1];
        }

        double const &
        output(std::size_t const &p_index) const {
            return o[o.size() - 1][p_index];
        }

        void
        randomizeWeights(unsigned int const &p_seed = time(nullptr)) {
            std::srand(p_seed);
            for (auto &mat : w)
                for (auto &vec : mat)
                    for (auto &val : vec)
                        val = weights_min + (weights_max - weights_min) * std::rand() / RAND_MAX;
            storing_loop = 0;
            step = 0;
            m = {w};
        }

        void
        remember(std::vector<double> const &p_pattern) {
            std::size_t
            layer = 0;
            for (std::size_t i = 0; i < layer_sizes[layer]; ++i) {
                o[layer][i] = p_pattern[i];
            }
            while (layer + 1 < layer_sizes.size()) {
                for (std::size_t i = 0; i < w[layer].size(); ++i) {
                    double
                    sum = 0.;
                    for (std::size_t j = 0; j < o[layer].size(); ++j) {
                        sum += w[layer][i][j] * o[layer][j];
                    }
                    o[layer + 1][i] = act(sum);
                }
                ++layer;
            }
        }

        bool
        saveHistory(std::string const &p_filename) const {
            return alg::save(p_filename, m);
        }

        bool
        saveWeights(std::string const &p_filename) const {
            return alg::save(p_filename, w);
        }

        bool
        saveMe(std::string const &p_filename) {
            std::ofstream ofs(p_filename + "-meta.dat");
            if (!ofs.is_open())
                return false;
            ofs << "Activation-Min: " << act.mn << std::endl
                << "Activation-Max: " << act.mx << std::endl
                << "Weights-Min:    " << weights_min << std::endl
                << "Weights-Max:    " << weights_max << std::endl
                << "Eta:            " << eta0 << std::endl
                << "Delta-Eta:      " << delta_eta << std::endl
                << "Eta-Halftime:   " << eta_halftime << std::endl
                << "Step:           " << step << std::endl
                << "Storing-Period: " << storing_period << std::endl
                << "Batch-Size:     " << batch_size << std::endl
                << "Layer-Sizes:    " << str(layer_sizes) << std::endl;
            ofs.close();
            return saveHistory(p_filename + "-history.dat") && saveWeights(p_filename + "-weights.dat");
        }

        Brain &
        setBatchSize(std::size_t const &p_batchSize) {
            batch_size = p_batchSize;
            batch_count = 0;
            return * this;
        }

        Brain &
        setFlatSpotEliminationOffset(double const &p_flat_spot_elimination_offset) {
            dact.flat_spot_elimination_offset = p_flat_spot_elimination_offset;
            return * this;
        }

        template <typename T>
        std::string
        str(std::vector<T> const &p_vec, std::size_t p_len = 0) const {
            std::size_t
            len = 0;
            typename std::vector<T>::const_iterator
            ci = p_vec.cbegin(),
            ce = p_vec.cend();
            std::stringstream
            ss;
            if (p_len < 1) {
                while (ci != ce) {
                    ss.str("");
                    ss << *ci;
                    len = ss.str().length();
                    if (p_len < len) {
                        p_len = len;
                    }
                    ++ci;
                }
            }
            ci = p_vec.cbegin();
            ss.str("");
            ss << std::setw(p_len + 1) << *ci;
            while (++ci != ce) {
                ss << std::setw(p_len + 1) << *ci;
            }
            return ss.str();
        }

        void
        teach(std::vector<double> const &p_teacher) {
            std::size_t
            layer = layer_sizes.size() - 2;
            for (std::size_t i = 0; i < d[layer].size(); ++i) {
                d[layer][i] = dact(o[layer + 1][i]) * (p_teacher[i] - o[layer + 1][i]);
            }
            while (0 < layer) {
                --layer;
                for (std::size_t i = 0; i < d[layer].size(); ++i) {
                    double
                    sum = 0.;
                    for (std::size_t j = 0; j < d[layer + 1].size(); ++j) {
                        sum += d[layer + 1][j] * w[layer + 1][j][i];
                    }
                    d[layer][i] = dact(o[layer + 1][i]) * sum;
                }
            }
            eta = eta0 * pow(2., -double(step) / eta_halftime);            
            double
            e = eta,
            q1 = 1. - pow(1. - adam_beta1, step+1.),
            q2 = 1. - pow(1. - adam_beta2, step+1.);
            for (layer = 0; layer < w.size(); ++layer) {
                for (std::size_t i = 0; i < w[layer].size(); ++i) {
                    for (std::size_t j = 0; j < w[layer][i].size(); ++j) {
                        double
                        dLdW = - d[layer][i] * o[layer][j];
                        
                        adam_m[layer][i][j] = (adam_beta1 * adam_m[layer][i][j] + (1. - adam_beta1) * dLdW) / q1;
                        adam_v[layer][i][j] = (adam_beta2 * adam_v[layer][i][j] + (1. - adam_beta2) * dLdW * dLdW) / q2;
                        
                        double
                        d_w_tmp = - e * adam_m[layer][i][j] / (sqrt(adam_v[layer][i][j]) + 1.e-5);
                        
                        //d_w[layer][i][j] = d_w_tmp;
                        w[layer][i][j] += d_w_tmp;
                    //    w[layer][i][j] -= e * dLdW;
                        
                    }
                }
                e *= delta_eta;
            }
            if (storing_period && storing_period < ++storing_loop) {
                storing_loop = 0;
                m.push_back(w);
            }
            ++step;
        }

        void
        teach_batch(std::vector<double> const &p_teacher, double const & p_alpha = .0, double const & p_beta = 0.) {
            // alpha = .1, // 25,   //0 ... 1
            // beta = .0;  //.005 ... .030
            ++batch_count;
            std::size_t
            layer = layer_sizes.size() - 2;
            for (std::size_t i = 0; i < d[layer].size(); ++i) {
                d[layer][i] = dact(o[layer + 1][i]) * (p_teacher[i] - o[layer + 1][i]);
            }
            while (0 < layer) {
                --layer;
                for (std::size_t i = 0; i < d[layer].size(); ++i) {
                    double
                    sum = 0.;
                    for (std::size_t j = 0; j < d[layer + 1].size(); ++j) {
                        sum += d[layer + 1][j] * w[layer + 1][j][i];
                    }
                    d[layer][i] = dact(o[layer + 1][i]) * sum;
                }
            }
            eta = eta0 * pow(2., -double(step) / eta_halftime);
            double
            e = eta;
            for (layer = 0; layer < w.size(); ++layer) {
                for (std::size_t i = 0; i < w[layer].size(); ++i) {
                    for (std::size_t j = 0; j < w[layer][i].size(); ++j) {
                        double
                        //d_w_tmp = e * ((1. - p_alpha) * d[layer][i] * o[layer][j] + p_alpha * (p_beta * w[layer][i][j] - d_w[layer][i][j]));
                        d_w_tmp = e * ((1. - p_alpha) * d[layer][i] * o[layer][j] + p_alpha * d_w[layer][i][j]) - p_beta * w[layer][i][j];
                        d_w[layer][i][j] = d_w_tmp;
                        s_w[layer][i][j] += d_w_tmp;
                        if (batch_size <= batch_count) {
                            w[layer][i][j] += s_w[layer][i][j];
                            s_w[layer][i][j] = 0;
                        }
                    }
                }
                e *= delta_eta;
            }
            if (batch_size <= batch_count) {
                batch_count = 0;
            }
            if (storing_period && storing_period < ++storing_loop) {
                storing_loop = 0;
                m.push_back(w);
            }
            ++step;
        }

        std::vector<std::vector<double>> const
        &weights(std::size_t const &p_layer) const {
            return w[p_layer];
        }

        std::vector<double> const
        &weights(std::size_t const &p_layer, std::size_t const &p_row) const {
            return w[p_layer][p_row];
        }

    private:

        bool
        fromFileInputStream(std::ifstream &p_ifs) {
            if (!alg::load(p_ifs, w)) {                
                return false;
            }
            layer_sizes.resize(w.size() + 1);
            layer_sizes[0] = w[0][0].size() - 1;
            for (std::size_t i = 0; i < w.size(); ++i) {
                layer_sizes[i + 1] = w[i].size();
            }
            o.resize(0);
            d.resize(0);
            for (std::size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                o.push_back(std::vector<double>(layer_sizes[i] + 1, 1.));
            }
            o.push_back(std::vector<double>(layer_sizes[layer_sizes.size() - 1], 0.));
            for (std::size_t i = 1; i < layer_sizes.size(); ++i) {
                d.push_back(std::vector<double>(layer_sizes[i]));
            }
            return true;
        }

        bool
        loadMe(std::string const &p_filename) {

            std::ifstream ifs(p_filename + "-meta.dat");

            if (!ifs.is_open()) {
                return false;
            }

            std::string
            line;

            double
            act_mn, act_mx, lweights_min, lweights_max, leta0, ldelta_eta, leta_halftime;

            std::size_t
            lstep, lstoring_period, lbatch_size;

            std::string
            dummy;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> act_mn;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> act_mx;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> lweights_min;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> lweights_max;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> leta0;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> ldelta_eta;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> leta_halftime;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> lstep;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> lstoring_period;
            } else {
                return false;
            }
            if (std::getline(ifs, line)) {
                std::istringstream iss(line);
                iss >> dummy >> lbatch_size;
            } else {
                return false;
            }
            ifs.close();
            this->act = Sig(act_mn, act_mx);
            this->act = DSig(act_mn, act_mx);
            this->weights_min = lweights_min;
            this->weights_max = lweights_max;
            this->eta0 = leta0;
            this->eta = this->eta0;
            this->delta_eta = ldelta_eta;
            this->eta_halftime = leta_halftime;
            this->step = lstep;
            this->storing_period = lstoring_period;
            this->batch_size = lbatch_size;
            this->batch_count = 0;
            loadHistory(p_filename + "-history.dat");
            std::ifstream ifsW(p_filename + "-weights.dat");
            if (ifsW.is_open()) {
                bool succ = fromFileInputStream(ifsW);
                ifsW.close();
                return succ;
            }
            return false;
        }
};

#endif // BRAIN_HPP
