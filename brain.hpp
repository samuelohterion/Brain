#ifndef BRAIN_HPP
#define BRAIN_HPP

#include <math.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

class Brain {

	private:

		class Sig {

			protected:

				double mn, mx;

			public:

				Sig(double const & p_min = 0, double const & p_max = 1.) :
				mn(p_min),
				mx(p_max) {

				}

				double operator()(double const & p_net) const {

					return mn + (mx - mn) / (1. + exp(-p_net));
				}
		};

		class DSig : public Sig {

			public:

				DSig(double const & p_min = 0., double const & p_max = 1.) :
				Sig(p_min, p_max) {

				}

				double operator()(double const & p_act) const {

					return (mx - p_act) * (p_act - mn) / (mx - mn);
					//return .0001 + (mx - p_act) * (p_act - mn) / (mx - mn);
				}
			};

	public:

		std::vector< std::size_t > const
		layer_sizes;

		Sig const  act;
		DSig const dact;

		double
		eta;

		std::vector< std::vector< double > >
		o, //utput
		d; //elta

		std::vector< std::vector< std::vector< double > > >
		w;

		std::vector< std::vector< std::vector< std::vector< double > > > >
		m;

		std::size_t
		outer_loop, inner_loop;

		std::size_t
		save_weights_every_n_loops;

	public:

		Brain( std::initializer_list< std::size_t > const & p_layer_sizes,
				double p_eta = .5,
				double const &  p_activation_min = 0., double const & p_activation_max = 1.,
				double const &  p_weights_min = -1., double const & p_weights_max = 1.,
				std::size_t const & p_seed = time(nullptr),
				std::size_t const & p_save_weights_every_n_loops = 0 ) :
		layer_sizes( p_layer_sizes.begin( ), p_layer_sizes.end( ) ),
		act(  p_activation_min, p_activation_max ),
		dact( p_activation_min, p_activation_max ),
		eta( p_eta ),
		save_weights_every_n_loops( p_save_weights_every_n_loops ) {

			configure( p_weights_min, p_weights_max, p_seed );
		}

		Brain( std::vector< std::size_t > const & p_layerSizes,
				double p_eta = .5,
				double const &  p_activation_min = 0., double const & p_activation_max = 1.,
				double const &  p_weights_min = -1., double const & p_weights_max = 1.,
				std::size_t const & p_seed = time(nullptr),
				std::size_t const & p_save_weights_every_n_loops = 0 ) :
		layer_sizes( p_layerSizes.begin( ), p_layerSizes.end( ) ),
		act(  p_activation_min, p_activation_max ),
		dact( p_activation_min, p_activation_max ),
		eta( p_eta ),
		save_weights_every_n_loops( p_save_weights_every_n_loops ) {

			configure( p_weights_min, p_weights_max, p_seed );
		}

		static std::vector< double >
		digitize( double const & p_x, double const & p_x0, double const & p_x1, std::size_t p_digits ) {

			double
			x = ( p_x - p_x0 ) / ( p_x1 - p_x0 );

			x =
				x < 0
					? 0
					: 1 < x
						? 1
						: x;

			std::vector< double >
			bit( p_digits );

			while ( 0 < p_digits ) {

				-- p_digits;

				if( x < .5 ) {

					bit[ p_digits ] = 0;
				}
				else {

					bit[ p_digits ] = 1;

					x -= .5;
				}

				x *= 2;
			}

			return bit;
		}

		static double
		analogize( std::vector< double > p_bits, double const & p_x0, double const & p_x1 ) {

			double
			ret = 0.,
			sum = .5;

			std::size_t
			i = p_bits.size( );

			while ( 0 < i ) {

				ret +=
					.5 < p_bits[ -- i ]
						? sum
						: 0.;
				sum *= .5;
			}

			return p_x0 + ( p_x1 - p_x0 ) * ret;
		}

		void
		configure( double const &  p_weights_min = 0., double const & p_weights_max = 1., std::size_t const & p_seed = std::time(nullptr) ) {

			for( std::size_t i = 0; i < layer_sizes.size( ) - 1; ++ i ) {

				o.push_back( std::vector< double >( layer_sizes[ i ] + 1, 1. ) );
			}

			o.push_back( std::vector< double >( layer_sizes[ layer_sizes.size() - 1 ], 0. ) );

			for( std::size_t i = 1; i < layer_sizes.size( ); ++ i ) {

				d.push_back( std::vector< double >( layer_sizes[ i ] ) );
			}

			for( std::size_t i = 0; i < layer_sizes.size( ) - 1; ++ i ) {

				std::size_t
				realNumberOfNeuoronsInLayer     = layer_sizes[ i + 1 ],
				realNumberOfNeuoronsInPrevLayer = layer_sizes[ i ] + 1;

				w.push_back( std::vector< std::vector< double > >( realNumberOfNeuoronsInLayer, std::vector< double >( realNumberOfNeuoronsInPrevLayer ) ) );
			}

			randomizeWeights( p_weights_min, p_weights_max, p_seed );
		}

		double
		error( std::vector< double > const & p_teacher ) const {

			std::size_t
			lyr = layer_sizes.size( ) - 1,
			to  = layer_sizes[ lyr ];

			double
			r = 0,
			tmp;

			for( std::size_t i = 0; i < to; ++ i ) {

				tmp = o[ lyr ][ i ] - p_teacher[ i ];

				r += tmp * tmp;
			}

			return sqrt( r );
		}

		double
		rms( ) const {

			std::size_t
			lyr = d.size( ) - 1,
			to  = d[ lyr ].size( );

			double
			r = 0;

			for( std::size_t i = 0; i < to; ++ i ) {

				r += d[ lyr ][ i ] * d[ lyr ][ i ];
			}

			return sqrt( r / to );
		}

		struct ERROR {

			std::size_t
			minId,
			maxId;

			double
			minVal,
			maxVal,
			sum;

			ERROR( ) :
			minId( 0 ),
			maxId( 0 ),
			sum( 0. ) {

			}

			std::string
			str( ) const {

				std::stringstream
				ss;

				ss
					<< "Error summary:" << std::endl
					<< "  min id   : " << this->minId << std::endl
					<< "  min val  : " << this->minVal << std::endl
					<< "  max id   : " << this->maxId << std::endl
					<< "  max val  : " << this->maxVal << std::endl
					<< "  err^2 tot: "
					<< this->sum << std::endl
					<< "  err tot: "
					<< sqrt( this->sum ) << std::endl;

				return ss.str( );
			}
		};

		ERROR
		errorTotal( std::vector< std::vector< double > > const & p_patterns, std::vector< std::vector< double > > const & p_teachers ) {

			ERROR
			err;

			std::size_t
			i = 0;

			remember( p_patterns[ i ] );

			double
			errVal = error( p_teachers[ i ] );

			err.sum = errVal;

			err.minId  = err.maxId = i;
			err.minVal = err.maxVal = errVal;

			while( ++ i < p_teachers.size( ) ) {

				remember( p_patterns[ i ] );

				errVal = error( p_teachers[ i ] );

				err.sum += errVal;

				if( errVal < err.minVal ) {

					err.minId  = i;
					err.minVal = errVal;
				}

				if( err.maxVal < errVal ) {

					err.maxId = i;
					err.maxVal = errVal;
				}
			}

			return err;
		}

		std::vector< std::vector< std::vector< std::vector< double > > > >
		& history( ) {

			return m;
		}

		std::vector< std::vector< std::vector< double > > > const
		& history( std::size_t const & p_id ) const {

			return m[ p_id ];
		}

		std::vector< double > const
		& input ( ) const {

			return o[ 0 ];
		}

		double const
		& input( std::size_t const & p_index ) const {

			return o[ 0 ][ p_index ];
		}

		void
		norm( ) {

			double
			maxOut = 0.;

			for ( std::size_t l = 0; l < w.size( ); ++ l ) {

				for ( std::size_t r = 0; r < w[ l ].size( ); ++ r ) {

					for ( std::size_t c = 0; c < w[ l ][ r ].size( ); ++ c ) {

						if( maxOut < abs( w[ l ][ r ][ c ] ) ) {

							maxOut = abs( w[ l ][ r ][ c ] );
						}
					}
				}
			}

			if( maxOut < 1e-6 ) {

				return;
			}

			for ( std::size_t l = 0; l < w.size( ); ++ l ) {

				for ( std::size_t r = 0; r < w[ l ].size( ); ++ r ) {

					for ( std::size_t c = 0; c < w[ l ][ r ].size( ); ++ c ) {

						if( maxOut < abs( w[ l ][ r ][ c ] ) ) {

							w[ l ][ r ][ c ] /= maxOut;
						}
					}
				}
			}
		}

		std::vector< double > const
		& output( ) const {

			return o[ o.size( ) - 1 ];
		}

		double const
		& output( std::size_t const & p_index ) const {

			return o[ o.size( ) - 1 ][ p_index ];
		}

		std::string
		str( std::vector< double > const & p_vec, std::size_t p_len = 0 ) const {

			std::size_t
			len    = 0;

			std::vector< double >::const_iterator
			ci = p_vec.cbegin( ),
			ce = p_vec.cend( );

			std::stringstream
			ss;

			if( p_len < 1 ) {

				while( ci != ce ) {

					ss.str( "" );

					ss << * ci;

					len = ss.str( ).length( );

					if( p_len < len ) {

						p_len = len;
					}

					++ ci;
				}
			}

			ci = p_vec.cbegin( );

			ss.str( "" );

			ss << std::setw( p_len + 1 ) << * ci;

			while( ++ ci != ce ) {

				ss << std::setw( p_len + 1 ) << * ci;
			}

			return ss.str( );
		}

		void
		randomizeWeights( double const &  p_weights_min = 0., double const & p_weights_max = 1., std::size_t const & p_seed = time( nullptr ) ) {

			srand( p_seed );

			for( auto & mat : w )

				for( auto & vec : mat )

					for( auto & val : vec )

						val = p_weights_min + ( p_weights_max - p_weights_min ) * std::rand( ) / RAND_MAX;

			outer_loop = 0;

			inner_loop = 0;

			m = { w };
		}

		void
		remember( std::vector< double > const & p_pattern ) {

			std::size_t
			layer = 0;

			for( std::size_t i = 0; i < layer_sizes[ layer ]; ++ i ) {

				o[ layer ][ i ] = p_pattern[ i ];
			}

			while( layer + 1 < layer_sizes.size( ) ) {

				for ( std::size_t i = 0; i < w[ layer ].size( ); ++ i ) {

					double
					sum = 0.;

					for ( std::size_t j = 0; j < o[ layer ].size( ); ++ j ) {

						sum += w[ layer ][ i ][ j ] * o[ layer ][ j ];
					}

					o[ layer + 1 ][ i ] = act( sum );
				}

				++ layer;
			}
		}

		void
		teach( std::vector< double > const & p_teacher ) {

			std::size_t
			layer = layer_sizes.size( ) - 2;

			for( std::size_t i = 0; i < d[ layer ].size( ); ++ i ) {

				d[ layer ][ i ] = dact( o[ layer + 1 ][ i ] ) * (o[ layer + 1 ][ i ] - p_teacher[ i ]);
			}

			while(0 < layer) {

				-- layer;

				for( std::size_t i = 0; i < d[ layer ].size( ); ++ i ) {

					double sum = 0.;

					for( std::size_t j = 0; j < d[ layer + 1 ].size( ); ++ j ) {

						sum += d[ layer + 1 ][ j ] * w[ layer + 1 ][ j ][ i ];
					}

					d[ layer ][ i ] = dact( o[ layer + 1 ][ i ] ) * sum;
				}
			}

			double
			e = eta;

			for( layer = 0; layer < w.size( ); ++ layer ) {

				for( std::size_t i = 0; i < w[ layer ].size( ); ++ i ) {

					for( std::size_t j = 0; j < w[ layer ][ i ].size( ); ++ j ) {

						w[ layer ][ i ][ j ] -= e * d[ layer ][ i ] * o[ layer ][ j ];
					}
				}

				//e *= .9;
			}

			++ outer_loop;
			++ inner_loop;

			if( ++ inner_loop == save_weights_every_n_loops ) {

				inner_loop = 0;

				m.push_back( w );
			}
		}
};

#endif // BRAIN_HPP
