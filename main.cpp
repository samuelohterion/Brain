#include <math.h>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include "brain.hpp"
#include "../AlgebraWithSTL/mlp.hpp"

using namespace std;
using namespace alg;

int
main( ) {

	// cout << "MLP\n---\n\n";
/*
	srand( 3 );//time( nullptr ) );

	Brain
	brain( { 2, 2, 1 }, .5, 0., 1. );

	std::vector< std::vector< double > >
	pattern = {{0,0},{0,1},{1,0},{1,1}},
	teacher = {{0},{1},{1},{0}};

	for( std::size_t i = 0; i < 10000; ++ i ) {

		std::size_t
		j = rand( ) & 0x03;

		brain.remember( pattern[ j ] );
		brain.teach( teacher[ j ] );
	}

	for( std::size_t i = 0; i < 4; ++ i ) {

		brain.remember( pattern[ i ] );

		std::cout << brain.str( brain.input( ), 1 ) << " => " << brain.str( brain.output( ) ) << std::endl;
	}
	*/
	MD
	x_y = mcnst< D >(64, 6),        // 64 x (3 + 3) bits: y = y2 y1 y0   x = x2 x1 x0
	x_times_y = mcnst< D >(64, 6);  // 64 x 6 bits as result of x * y

	for (UI y = 0; y < 8; ++ y) {

		for (UI x = 0; x < 8; ++ x) {

			UI
			j = (y << 3) + x;

			for (UI i = 0; i < 6; ++ i) {

				x_y      [j][5 - i] = (j         & (1ul << i)) == (1ul << i);
				x_times_y[j][5 - i] = ((y * x) & (1ul << i)) == (1ul << i);
			}
		}
	}

	print("x_y", ~ x_y);
	print("x_times_y", ~ x_times_y);
	srand(1);

	MLP
	mlp({6, 8, 12, 8, 6}, .2, 0., 1., -1., +1., 1);

	std::vector< std::vector< std::vector< double > > >
	mlp_weights = mlp.weights();

	UI
	loops = 160000,
	lloop = loops / 10,
	ll    = 0;

	for(UI loop = 1; loop <= loops; ++ loop) {

		UI
		pId = rand() & 0x3f;

		mlp.remember(x_y[pId]);

		mlp.teach(x_times_y[pId]);

//		if((loop == 1) || (loop == 10) || (loop == 1e2) || (loop == 1e3) || (loop == 1e4) || (loop == 1e5) || (loop == 1e6)) {
		if(++ ll == lloop) {

			ll = 0;

			print("loop", loop);
			print("sse:", mlp.rms());
			std::cout
				<< " --------------------------------------------- " << std::endl
				<< "| Bits:                     0   correct 0     |" << std::endl
				<< "| Bits:                     1 incorrect 0     |" << std::endl
				<< "| Bits:                     2 incorrect 1     |" << std::endl
				<< "| Bits:                     3   correct 1     |" << std::endl
				<< " ---------- ---------------------------------- " << std::endl
				<< "| Y        |  X        ==>  Z = X * Y         |" << std::endl
				<< "| 2  1  0  |  2  1  0       5  4  3  2  1  0  |" << std::endl
				<< " ----------+---------------------------------- " << std::endl;

			for(UI i = 0; i < 64; ++ i) {

				mlp.remember(x_y[i]);

				std::cout
					<< "| "
					<< sub(x_y[i], 0, 3)
					<< "|  "
					<< sub(x_y[i], 3, 3)
					<< "==>  "
					<< round((2. * mlp.output() + x_times_y[i]))
					<< "|"
					<< std::endl;
			}

			std::cout
				<< " ---------- ---------------------------------- " << std::endl << std::endl;
		}
	}

	print("x_y", ~ x_y);
	print("x_times_y", ~ x_times_y);
	srand(1);

	Brain
	brain({6, 8, 12, 8, 6}, .2, 0., 1., -1., +1., 1);

	std::vector< std::vector< std::vector< double > > >
	brain_weights = brain.w;

	loops = 160000,
	lloop = loops / 10,
	ll    = 0;

	for(UI loop = 1; loop <= loops; ++ loop) {

		UI
		pId = rand() & 0x3f;

		brain.remember(x_y[pId]);

		brain.teach(x_times_y[pId]);

//		if((loop == 1) || (loop == 10) || (loop == 1e2) || (loop == 1e3) || (loop == 1e4) || (loop == 1e5) || (loop == 1e6)) {
		if(++ ll == lloop) {

			ll = 0;

			print("loop", loop);
			print("sse:", brain.rms());
			std::cout
				<< " --------------------------------------------- " << std::endl
				<< "| Bits:                     0   correct 0     |" << std::endl
				<< "| Bits:                     1 incorrect 0     |" << std::endl
				<< "| Bits:                     2 incorrect 1     |" << std::endl
				<< "| Bits:                     3   correct 1     |" << std::endl
				<< " ---------- ---------------------------------- " << std::endl
				<< "| Y        |  X        ==>  Z = X * Y         |" << std::endl
				<< "| 2  1  0  |  2  1  0       5  4  3  2  1  0  |" << std::endl
				<< " ----------+---------------------------------- " << std::endl;

			for(UI i = 0; i < 64; ++ i) {

				brain.remember(x_y[i]);

				std::cout
					<< "| "
					<< sub(x_y[i], 0, 3)
					<< "|  "
					<< sub(x_y[i], 3, 3)
					<< "==>  "
					<< round((2. * brain.output() + x_times_y[i]))
					<< "|"
					<< std::endl;
			}

			std::cout
				<< " ---------- ---------------------------------- " << std::endl << std::endl;
		}
	}

	print("mlp_weights", mlp_weights);
	print("brain_weights", brain_weights);


/*
	MLP
	ramp( { 2, 3, 1 }, .1, -2, +2 );

	std::vector< double >
	inp = { 0, 0 },
	out = { 0 };

	for( std::size_t i = 0; i < 10000; ++ i ) {

		inp[ 0 ] = 1. * rand( ) / RAND_MAX - .5;
		inp[ 1 ] = 1. * rand( ) / RAND_MAX - .5;
		out[ 0 ] = inp[ 0 ] + inp[ 1 ];

		ramp.remember( inp );
		ramp.teach( out );
	}

	std::vector< std::vector< double > >
	pMem = mcnst< double >( 201, 201 );

	for( std::size_t i = 0; i <= 200; ++ i ) {

		for( std::size_t j = 0; j <= 200; ++ j ) {

			inp[ 0 ] = .005 * i - .5;
			inp[ 1 ] = .005 * j - .5;

			ramp.remember( inp );

			pMem[ i ][ j ] = ramp.output( 0 ) - inp[ 0 ] - inp[ 1 ] ;

			std::cout << inp[ 0 ] << " + " << inp[ 1 ] << " = " << ramp.output( 0 ) << "  ð = " << ( ramp.output( 0 ) - inp[ 0 ] - inp[ 1 ] ) <<  std::endl;
		}

		std::cout << std::endl;
	}

	save( "ramp.txt", pMem );

	MLP
	ramp2( { 8 + 8, 15, 12, 9 }, 1., 0, 1 );

	for( std::size_t i = 0; i < 0x10000 * 100; ++ i ) {

		double
		a = ( rand( ) & 0xff ),
		b = ( rand( ) & 0xff ),
		c = a + b;

		std::vector< double >
		aBits = MLP::digitize( a, 0, 0x100, 8 ),
		bBits = MLP::digitize( b, 0, 0x100, 8 ),
		outBits = MLP::digitize( c, 0, 0x200, 9 ),
		inBits = std::vector< double >( 16 );

		std::copy( aBits.cbegin( ), aBits.cend( ), inBits.begin( ) + 0 );
		std::copy( bBits.cbegin( ), bBits.cend( ), inBits.begin( ) + 8 );

		ramp2.remember( inBits );
		ramp2.teach( outBits );
	}

	std::vector< std::vector< double > >
	rampMem( 0x100, std::vector< double >( 0x100 ) );

	for( std::size_t i = 0; i < 0x100; ++ i ) {

		for( std::size_t j = 0; j < 0x100; ++ j ) {

			double
			a = i,
			b = j,
			c = a + b;

			std::vector< double >
			aBits = MLP::digitize( a, 0, 0x100, 8 ),
			bBits = MLP::digitize( b, 0, 0x100, 8 ),
			outBits = MLP::digitize( c, 0, 0x200, 9 ),
			inBits = std::vector< double >( 16 );

			std::copy( aBits.cbegin( ), aBits.cend( ), inBits.begin( ) + 0 );
			std::copy( bBits.cbegin( ), bBits.cend( ), inBits.begin( ) + 8 );

			ramp2.remember( inBits );
			rampMem[ i ][ j ] = MLP::analogize( ramp2.output( ), 0, 0x200 ) - c;
		}
	}

	save( "ramp2.txt", rampMem );
*/
/*

	Brain
	m( { 1, 16, 8, 2 }, .1, -1, 1 );

	std::vector< double >
	x( 1 ),
	outX( 2 );

	for( std::size_t k = 0; k < 10; ++ k ) {

		for( std::size_t i = 0; i <= 10000000; ++ i ) {

			x[ 0 ] = 2. * rand( ) / RAND_MAX - 1.;

			outX[ 0 ] = + x[ 0 ];
			outX[ 1 ] = sin( 8. * 3.1415 * x[ 0 ] );

			m.remember( x );
			m.teach( outX );
		}

		for( std::size_t i = 0; i <= 100; ++ i ) {

			x[ 0 ] = .02 * i - 1.;

			m.remember( x );

	//		print( "x", x );
	//		print( "y", round( m.output( ), 3 ) );

			std::cout << x[ 0 ] << "\t" << round( m.output( 0 ), 3 ) << "\t" << round( m.output( 1 ), 3 ) << "\t" << k << std::endl;
		}
	}
*/
	return 0;
}
