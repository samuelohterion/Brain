#ifndef BRAIN_HPP
#define BRAIN_HPP

#include <math.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <deque>
//#include <numeric>
#include "../AlgebraWithSTL/algebra.hpp"
//#include "algebra.hpp" // for buildBrain.m

template< typename T = int >
class QueueSum : public std::deque< T > {

	private:

		T
		__sum;

	public:

		QueueSum(std::size_t const & p_size) :
		std::deque< T >(p_size),
		__sum(0) {

		}

		QueueSum< T > &
		add(T const & p_value) {

			__sum -= this->back();
			this->pop_back();
			this->push_front(p_value);
			__sum += p_value;

			return * this;
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
            double mn, mx, dst;

        public:
            Sig(double const &p_min = 0, double const &p_max = 1.)
                : mn(p_min)
                , mx(p_max)
                , dst(mx - mn)
            {}

            double operator()(double const &p_net) const { return mn + dst / (1. + exp(-p_net)); }
        };

        class DSig : public Sig
        {
        public:
            double mn, mx, dstRec;

        public:
            DSig(double const &p_min = 0., double const &p_max = 1.)
                : mn(p_min)
                , mx(p_max)
                , dstRec(1 / (mx - mn))
            {}

            double operator()(double const &p_act) const
            {
                //return (mx - p_act) * (p_act - mn) / (mx - mn);
                return .001 + (mx - p_act) * (p_act - mn) * dstRec;
            }
        };

    public:
        std::vector<std::size_t> layer_sizes;

        Sig act;

        DSig dact;

        double eta0, eta, eta_halftime, delta_eta, weights_min, weights_max;

        std::size_t batch_size, batch_count;

        std::vector<std::vector<double> > o, //utput
            d;                               //elta

        std::vector<std::vector<std::vector<double> > > w, //eights
            d_w, s_w;

        std::vector<std::vector<std::vector<std::vector<double> > > > m; //emory

        std::size_t outer_loop, inner_loop, step;

        std::size_t storing_period;

    public:
        Brain(std::initializer_list<std::size_t> const &p_layer_sizes,
              double const &p_eta = .5,
              double const &p_eta_halftime = 1000,
              double const &p_delta_eta = .8,
              double const &p_activation_min = 0.,
              double const &p_activation_max = 1.,
              double const &p_weights_min = -1.,
              double const &p_weights_max = 1.,
              std::size_t const &p_seed = time(nullptr),
              std::size_t const &p_weights_to_history_storing_period = 0,
              std::size_t const &p_batch_size = 0)
            : layer_sizes(p_layer_sizes.begin(), p_layer_sizes.end())
            , act(p_activation_min, p_activation_max)
            , dact(p_activation_min, p_activation_max)
            , eta0(p_eta)
            , eta(eta0)
            , eta_halftime(p_eta_halftime)
            , delta_eta(p_delta_eta)
            , weights_min(p_weights_min)
            , weights_max(p_weights_max)
            , batch_size(p_batch_size)
            , batch_count(0)
            , storing_period(p_weights_to_history_storing_period)
        {
            configure(p_weights_min, p_weights_max, p_seed);
        }

        Brain(std::vector<std::size_t> const &p_layerSizes,
              double const &p_eta = .5,
              double const &p_eta_halftime = 1000,
              double const &p_delta_eta = .8,
              double const &p_activation_min = 0.,
              double const &p_activation_max = 1.,
              double const &p_weights_min = -1.,
              double const &p_weights_max = 1.,
              std::size_t const &p_seed = time(nullptr),
              std::size_t const &p_weights_to_history_storing_period = 0,
              std::size_t const &p_batch_size = 0)
            : layer_sizes(p_layerSizes.begin(), p_layerSizes.end())
            , act(p_activation_min, p_activation_max)
            , dact(p_activation_min, p_activation_max)
            , eta0(p_eta)
            , eta(eta0)
            , eta_halftime(p_eta_halftime)
            , delta_eta(p_delta_eta)
            , weights_min(p_weights_min)
            , weights_max(p_weights_max)
            , batch_size(p_batch_size)
            , batch_count(0)
            , storing_period(p_weights_to_history_storing_period)
        {
            configure(p_weights_min, p_weights_max, p_seed);
        }

        Brain(std::string const &p_filename) { loadMe(p_filename); }

        static std::vector<double> digitize(double const &p_x,
                                            double const &p_x0,
                                            double const &p_x1,
                                            std::size_t p_digits)
        {
            double x = (p_x - p_x0) / (p_x1 - p_x0);

            x = x < 0 ? 0 : 1 < x ? 1 : x;

            std::vector<double> bit(p_digits);

            while (0 < p_digits) {
                --p_digits;

                if (x < .5) {
                    bit[p_digits] = 0;
                } else {
                    bit[p_digits] = 1;

                    x -= .5;
                }

                x *= 2;
            }

            return bit;
        }

        static double analogize(std::vector<double> p_bits, double const &p_x0, double const &p_x1)
        {
            double sum = 0., part = .5;

            std::size_t i = p_bits.size();

            while (0 < i) {
                sum += .5 < p_bits[--i] ? part : 0.;
                part *= .5;
            }

            return p_x0 + (p_x1 - p_x0) * sum;
        }

        Brain &configure(double const &p_weights_min = 0.,
                         double const &p_weights_max = 1.,
                         std::size_t const &p_seed = std::time(nullptr))
        {
            weights_min = p_weights_min;
            weights_max = p_weights_max;

            for (std::size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                o.push_back(std::vector<double>(layer_sizes[i] + 1, 1.));
            }

            o.push_back(std::vector<double>(layer_sizes[layer_sizes.size() - 1], 0.));

            for (std::size_t i = 1; i < layer_sizes.size(); ++i) {
                d.push_back(std::vector<double>(layer_sizes[i]));
            }

            for (std::size_t i = 0; i < layer_sizes.size() - 1; ++i) {
                std::size_t realNumberOfNeuoronsInLayer = layer_sizes[i + 1],
                            realNumberOfNeuoronsInPrevLayer = layer_sizes[i] + 1;

                w.push_back(std::vector<std::vector<double> >(realNumberOfNeuoronsInLayer,
                                                              std::vector<double>(
                                                                  realNumberOfNeuoronsInPrevLayer)));
                //                w_tmp.push_back(std::vector< std::vector< double > >(realNumberOfNeuoronsInLayer, std::vector< double >(realNumberOfNeuoronsInPrevLayer)));
                d_w.push_back(
                    std::vector<std::vector<double> >(realNumberOfNeuoronsInLayer,
                                                      std::vector<double>(
                                                          realNumberOfNeuoronsInPrevLayer))),
                    s_w.push_back(
                        std::vector<std::vector<double> >(realNumberOfNeuoronsInLayer,
                                                          std::vector<double>(
                                                              realNumberOfNeuoronsInPrevLayer)));
            }

            randomizeWeights(p_seed);

            return *this;
        }

        double error(std::vector<double> const &p_teacher) const
        {
            std::size_t lyr = layer_sizes.size() - 1, to = layer_sizes[lyr];

            double r = 0, tmp;

            for (std::size_t i = 0; i < to; ++i) {
                tmp = o[lyr][i] - p_teacher[i];

                r += tmp * tmp;
            }

            return sqrt(r);
        }

        double rms() const
        {
            std::size_t lyr = d.size() - 1, to = d[lyr].size();

            double r = 0;

            for (std::size_t i = 0; i < to; ++i) {
                r += d[lyr][i] * d[lyr][i];
            }

            return sqrt(r / to);
        }

        struct ERROR
        {
            std::size_t minId, maxId;

            double minVal, maxVal, sum;

            ERROR()
                : minId(0)
                , maxId(0)
                , sum(0.)
            {}

            std::string str() const
            {
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
        errorTotal(std::vector<std::vector<double> > const &p_patterns,
                   std::vector<std::vector<double> > const &p_teachers)
        {
            ERROR
            err;

            std::size_t i = 0;

            remember(p_patterns[i]);

            double errVal = error(p_teachers[i]);

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

        std::vector<std::vector<std::vector<std::vector<double> > > > &history() { return m; }

        bool loadHistory(std::string const &p_filename) { return alg::load(p_filename, m); }

        bool loadWeights(std::string const &p_filename) { return alg::load(p_filename, w); }

        std::vector<std::vector<std::vector<double> > > const &history(std::size_t const &p_id) const
        {
            return m[p_id];
        }

        std::vector<std::vector<double> > const &history(std::size_t const &p_id,
                                                         std::size_t const &p_layer) const
        {
            return m[p_id][p_layer];
        }

        std::vector<double> const &history(std::size_t const &p_id,
                                           std::size_t const &p_layer,
                                           std::size_t const &p_row) const
        {
            return m[p_id][p_layer][p_row];
        }

        std::vector<double> const &input() const { return o[0]; }

        double const &input(std::size_t const &p_index) const { return o[0][p_index]; }

        void norm()
        {
            double maxOut = 0.;

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

        std::vector<double> const &output() const { return o[o.size() - 1]; }

        double const &output(std::size_t const &p_index) const { return o[o.size() - 1][p_index]; }

        void randomizeWeights(std::size_t const &p_seed = time(nullptr))
        {
            srand(p_seed);

            for (auto &mat : w)

                for (auto &vec : mat)

                    for (auto &val : vec)

                        val = weights_min + (weights_max - weights_min) * std::rand() / RAND_MAX;

            outer_loop = 0;

            inner_loop = 0;

            step = 0;

            m = {w};
        }

        void remember(std::vector<double> const &p_pattern)
        {
            std::size_t layer = 0;

            for (std::size_t i = 0; i < layer_sizes[layer]; ++i) {
                o[layer][i] = p_pattern[i];
            }

            while (layer + 1 < layer_sizes.size()) {
                for (std::size_t i = 0; i < w[layer].size(); ++i) {
                    double sum = 0.;

                    for (std::size_t j = 0; j < o[layer].size(); ++j) {
                        sum += w[layer][i][j] * o[layer][j];
                    }

                    o[layer + 1][i] = act(sum);
                }

                ++layer;
            }
        }

        bool saveHistory(std::string const &p_filename) const { return alg::save(p_filename, m); }

        bool saveWeights(std::string const &p_filename) const { return alg::save(p_filename, w); }

        bool saveMe(std::string const &p_filename)
        {
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

            return saveHistory(p_filename + "-history.dat")
                   && saveWeights(p_filename + "-weights.dat");
        }

        void setBatchSize(std::size_t const &p_batchSize)
        {
            batch_size = p_batchSize;
            batch_count = 0;
        }

        template<typename T>
        std::string str(std::vector<T> const &p_vec, std::size_t p_len = 0) const
        {
            std::size_t len = 0;

            typename std::vector<T>::const_iterator ci = p_vec.cbegin(), ce = p_vec.cend();

            std::stringstream ss;

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

        void teach(std::vector<double> const &p_teacher)
        {
            std::size_t layer = layer_sizes.size() - 2;

            for (std::size_t i = 0; i < d[layer].size(); ++i) {
                //                d[layer][i] = dact(o[layer + 1][i]) *
                //                (o[layer + 1][i] - p_teacher[i]);
                d[layer][i] = dact(o[layer + 1][i]) * (p_teacher[i] - o[layer + 1][i]);
            }

            while (0 < layer) {
                --layer;

                for (std::size_t i = 0; i < d[layer].size(); ++i) {
                    double sum = 0.;

                    for (std::size_t j = 0; j < d[layer + 1].size(); ++j) {
                        sum += d[layer + 1][j] * w[layer + 1][j][i];
                    }

                    d[layer][i] = dact(o[layer + 1][i]) * sum;
                }
            }

            eta = eta0 * pow(2., -double(step) / eta_halftime);

            double e = eta;

            double alpha = .1, // 25,   //0 ... 1
                beta = .0;     //.005 ... .030

            for (layer = 0; layer < w.size(); ++layer) {
                for (std::size_t i = 0; i < w[layer].size(); ++i) {
                    for (std::size_t j = 0; j < w[layer][i].size(); ++j) {
                        // w[layer][i][j] -= e * d[layer][i] * o[layer][j];
                        // w_tmp[layer][i][j] = w[layer][i][j];

                        double wTmp = w[layer][i][j];

                        double d_w_tmp = e
                                         * ((1. - alpha) * d[layer][i] * o[layer][j]
                                            + alpha * (d_w[layer][i][j] - beta * wTmp));

                        d_w[layer][i][j] = d_w_tmp;
                        w[layer][i][j] += d_w_tmp;
                    }
                }

                e *= delta_eta;
            }

            if (storing_period && storing_period < ++inner_loop) {
                inner_loop = 0;

                ++outer_loop;

                m.push_back(w);
            }

            ++step;
        }

        void teach_batch(std::vector<double> const &p_teacher)
        {
            ++batch_count;

            std::size_t layer = layer_sizes.size() - 2;

            for (std::size_t i = 0; i < d[layer].size(); ++i) {
                //                d[layer][i] = dact(o[layer + 1][i]) *
                //                (o[layer + 1][i] - p_teacher[i]);
                d[layer][i] = dact(o[layer + 1][i]) * (p_teacher[i] - o[layer + 1][i]);
            }

            while (0 < layer) {
                --layer;

                for (std::size_t i = 0; i < d[layer].size(); ++i) {
                    double sum = 0.;

                    for (std::size_t j = 0; j < d[layer + 1].size(); ++j) {
                        sum += d[layer + 1][j] * w[layer + 1][j][i];
                    }

                    d[layer][i] = dact(o[layer + 1][i]) * sum;
                }
            }

            eta = eta0 * pow(2., -double(step) / eta_halftime);

            double e = eta;

            double alpha = .1; // 25,   //0 ... 1
            double beta = .0;  //.005 ... .030

            for (layer = 0; layer < w.size(); ++layer) {
                for (std::size_t i = 0; i < w[layer].size(); ++i) {
                    for (std::size_t j = 0; j < w[layer][i].size(); ++j) {
                        // w[layer][i][j] -= e * d[layer][i] * o[layer][j];
                        // w_tmp[layer][i][j] = w[layer][i][j];

                        double wTmp = w[layer][i][j];
                        double d_w_tmp = e
                                         * ((1. - alpha) * d[layer][i] * o[layer][j]
                                            + alpha * (d_w[layer][i][j] - beta * wTmp));

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

            if (storing_period && storing_period < ++inner_loop) {
                inner_loop = 0;

                ++outer_loop;

                m.push_back(w);
            }

            ++step;
        }

        std::vector<std::vector<double> > const &weights(std::size_t const &p_layer) const
        {
            return w[p_layer];
        }

        std::vector<double> const &weights(std::size_t const &p_layer,
                                           std::size_t const &p_row) const
        {
            return w[p_layer][p_row];
        }

    private:
        bool fromFileInputStream(std::ifstream &p_ifs)
        {
            if (!alg::load(p_ifs, w))

                return false;

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

        bool loadMe(std::string const &p_filename)
        {
            std::ifstream ifs(p_filename + "-meta.dat");

            if (!ifs.is_open())

                return false;

            std::string line;

            double act_mn, act_mx, weights_min, weights_max, eta0, delta_eta, eta_halftime;

            std::size_t step, storing_period, batch_size;

            std::string dummy;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> act_mn;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> act_mx;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> weights_min;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> weights_max;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> eta0;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> delta_eta;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> eta_halftime;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> step;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> storing_period;
            } else
                return false;

            if (std::getline(ifs, line)) {
                std::istringstream iss(line);

                iss >> dummy >> batch_size;
            } else
                return false;

            ifs.close();

            this->act = Sig(act_mn, act_mx);
            this->act = DSig(act_mn, act_mx);
            this->weights_min = weights_min;
            this->weights_max = weights_max;
            this->eta0 = eta0;
            this->eta = this->eta0;
            this->delta_eta = delta_eta;
            this->eta_halftime = eta_halftime;
            this->step = step;
            this->storing_period = storing_period;
            this->batch_size = batch_size;

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
