
/* Standard C includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <stdio.h>
#include <smmintrin.h>

#include <stdio.h>
#include <stdint.h>

#include "hog.h"

typedef union suf32
{
    int i;
    unsigned u;
    float f;
}
suf32;

int histograms_of_gradient_directions_sse(unsigned char* src, unsigned char* bin0_ptr, unsigned char* bin1_ptr, unsigned char* bin2_ptr, unsigned char* bin3_ptr, unsigned char* bin4_ptr, unsigned char* bin5_ptr, unsigned char* bin6_ptr, unsigned char* bin7_ptr, unsigned char* bin8_ptr, unsigned char* bin9_ptr, unsigned char* bin10_ptr, unsigned char* bin11_ptr, unsigned char* bin12_ptr, unsigned char* bin13_ptr, unsigned char* bin14_ptr, unsigned char* bin15_ptr, int width, int height, int ksize )
{
	// init variables
        int* magnitudePtr = (int*) malloc(height*width*2*sizeof(int));
        uint16_t* binIndexPtr = (uint16_t*) malloc(height*width*2*sizeof(uint16_t));

	// dont change !! otherwise histograms will not be aligned
        int nbins = 16;

	int r = (ksize+1)/2;

	// variables needed for fast atan computation, taken from a corresponding OpenCV modules core/src/mathfuncs.cpp
	float angleScale = (float)(nbins/3.141592f/2.0f);

	float atan2_p1 = 0.9997878412794807f;
	float atan2_p3 = -0.3258083974640975f;
	float atan2_p5 = 0.1555786518463281f;
	float atan2_p7 = -0.04432655554792128f;

	double dblEpsilon = 2.2204460492503131e-16;

	__m128 phalf = _mm_set1_ps(0.5f), nhalf = _mm_set1_ps(-0.5f), fzero = _mm_setzero_ps();
        __m128 _angleScale = _mm_set1_ps(angleScale), fone = _mm_set1_ps(1.0f);
        __m128i ione = _mm_set1_epi32(1), _nbins = _mm_set1_epi32(nbins), _nbinsToCompare = _mm_set1_epi32(nbins-1), izero = _mm_setzero_si128();

	suf32 iabsmask; iabsmask.i = 0x7fffffff;
	__m128 eps = _mm_set1_ps((float)dblEpsilon), absmask = _mm_set1_ps(iabsmask.f);
	__m128 _90 = _mm_set1_ps(1.570796f), _180 = _mm_set1_ps(3.141592f), _360 = _mm_set1_ps(6.283072f);

	__m128 z = _mm_setzero_ps();
	__m128 p1 = _mm_set1_ps(atan2_p1), p3 = _mm_set1_ps(atan2_p3);
	__m128 p5 = _mm_set1_ps(atan2_p5), p7 = _mm_set1_ps(atan2_p7);

	for (int i=0; i < height - 2; i++)
	{
		for (int j=0 ; j<width-4 || (i != height-3 && j < width-2); j += 4)
		{
			int i2 = (i*width+j)*2;

			// sobel x
			__m128i _sobel0 =  _mm_loadl_epi64((__m128i*)(src + i*width+j));
			__m128i _sobel1 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+1));
			__m128i _sobel2 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+2));
			__m128i _sobel3 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+2*width));
			__m128i _sobel4 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+1+2*width));
			__m128i _sobel5 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+2+2*width));
			
			// store as integers
			_sobel0 = _mm_unpacklo_epi8(_sobel0,izero);
			_sobel0 = _mm_unpacklo_epi16(_sobel0,izero);
			_sobel1 = _mm_unpacklo_epi8(_sobel1,izero);
			_sobel1 = _mm_unpacklo_epi16(_sobel1,izero);
			_sobel2 = _mm_unpacklo_epi8(_sobel2,izero);
			_sobel2 = _mm_unpacklo_epi16(_sobel2,izero);
			_sobel3 = _mm_unpacklo_epi8(_sobel3,izero);
			_sobel3 = _mm_unpacklo_epi16(_sobel3,izero);
			_sobel4 = _mm_unpacklo_epi8(_sobel4,izero);
			_sobel4 = _mm_unpacklo_epi16(_sobel4,izero);
			_sobel5 = _mm_unpacklo_epi8(_sobel5,izero);
			_sobel5 = _mm_unpacklo_epi16(_sobel5,izero);

			_sobel0 = _mm_add_epi32(_sobel0,_sobel1);
			_sobel0 = _mm_add_epi32(_sobel0,_sobel1);
			_sobel0 = _mm_add_epi32(_sobel0,_sobel2);

			_sobel0 = _mm_sub_epi32(_sobel0,_sobel3);
			_sobel0 = _mm_sub_epi32(_sobel0,_sobel4);
			_sobel0 = _mm_sub_epi32(_sobel0,_sobel4);
			_sobel0 = _mm_sub_epi32(_sobel0,_sobel5);

			__m128 x = _mm_cvtepi32_ps(_sobel0);

			// sobel y
			_sobel0 =  _mm_loadl_epi64((__m128i*)(src + i*width+j));
			_sobel1 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+width));
			_sobel2 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+2*width));
			_sobel3 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+2));
			_sobel4 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+2+width));
			_sobel5 =  _mm_loadl_epi64((__m128i*)(src + i*width+j+2+2*width));

			_sobel0 = _mm_unpacklo_epi8(_sobel0,izero);
			_sobel0 = _mm_unpacklo_epi16(_sobel0,izero);
			_sobel1 = _mm_unpacklo_epi8(_sobel1,izero);
			_sobel1 = _mm_unpacklo_epi16(_sobel1,izero);
			_sobel2 = _mm_unpacklo_epi8(_sobel2,izero);
			_sobel2 = _mm_unpacklo_epi16(_sobel2,izero);
			_sobel3 = _mm_unpacklo_epi8(_sobel3,izero);
			_sobel3 = _mm_unpacklo_epi16(_sobel3,izero);
			_sobel4 = _mm_unpacklo_epi8(_sobel4,izero);
			_sobel4 = _mm_unpacklo_epi16(_sobel4,izero);
			_sobel5 = _mm_unpacklo_epi8(_sobel5,izero);
			_sobel5 = _mm_unpacklo_epi16(_sobel5,izero);

			_sobel0 = _mm_add_epi32(_sobel0,_sobel1);
			_sobel0 = _mm_add_epi32(_sobel0,_sobel1);
			_sobel0 = _mm_add_epi32(_sobel0,_sobel2);

			_sobel0 = _mm_sub_epi32(_sobel0,_sobel3);
			_sobel0 = _mm_sub_epi32(_sobel0,_sobel4);
			_sobel0 = _mm_sub_epi32(_sobel0,_sobel4);
			_sobel0 = _mm_sub_epi32(_sobel0,_sobel5);

			__m128 y = _mm_cvtepi32_ps(_sobel0);

			// angle (arctan approx, give 0 to 2*Pi)
			__m128 ax = _mm_and_ps(x, absmask), ay = _mm_and_ps(y, absmask);
			__m128 mask = _mm_cmplt_ps(ax, ay);
			__m128 tmin = _mm_min_ps(ax, ay), tmax = _mm_max_ps(ax, ay);
		
			__m128 c = _mm_div_ps(tmin, _mm_add_ps(tmax, eps));

			__m128 c2 = _mm_mul_ps(c, c);
			__m128 _angle = _mm_mul_ps(c2, p7);
			_angle = _mm_mul_ps(_mm_add_ps(_angle, p5), c2);
			_angle = _mm_mul_ps(_mm_add_ps(_angle, p3), c2);
			_angle = _mm_mul_ps(_mm_add_ps(_angle, p1), c);

			__m128 b = _mm_sub_ps(_90, _angle);
			_angle = _mm_xor_ps(_angle, _mm_and_ps(_mm_xor_ps(_angle, b), mask));

			b = _mm_sub_ps(_180, _angle);
			mask = _mm_cmplt_ps(x, z);
			_angle = _mm_xor_ps(_angle, _mm_and_ps(_mm_xor_ps(_angle, b), mask));

			b = _mm_sub_ps(_360, _angle);
			mask = _mm_cmplt_ps(y, z);
			_angle = _mm_xor_ps(_angle, _mm_and_ps(_mm_xor_ps(_angle, b), mask));

			// magnitude
			__m128 _mag = _mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y));
			_mag = _mm_sqrt_ps(_mag);

			// compute gradient
			_angle = _mm_mul_ps(_angleScale, _angle); // 0 - 15.99

			__m128i _hidx = _mm_cvttps_epi32(_angle); // 0 - 15

			__m128 delta = _mm_sub_ps(_angle, _mm_cvtepi32_ps(_hidx)); // 0.3

			__m128 ft0 = _mm_mul_ps(_mag, _mm_sub_ps(fone, delta));
			__m128 ft1 = _mm_mul_ps(_mag, delta);
			__m128 ft2 = _mm_unpacklo_ps(ft0, ft1);
			__m128 ft3 = _mm_unpackhi_ps(ft0, ft1);

			_mm_storeu_si128((__m128i*)(magnitudePtr + i2), _mm_cvttps_epi32(ft2));
			_mm_storeu_si128((__m128i*)(magnitudePtr + i2 + 4), _mm_cvttps_epi32(ft3));

			__m128i it0 = _mm_packs_epi32(_hidx, izero);
			_hidx = _mm_add_epi32(ione, _hidx);

			__m128i bin_mask = _mm_cmplt_epi32(_nbinsToCompare, _hidx);
			_hidx = _mm_sub_epi32(_hidx, _mm_and_si128(bin_mask, _nbins));

			__m128i it1 = _mm_packs_epi32(_hidx, izero);
		
			_mm_storeu_si128((__m128i*)(binIndexPtr + i2), _mm_unpacklo_epi16(it0, it1));
		}
		
		if( i == height-3 )
		{
			memset( magnitudePtr+2*(i*width+width-4), 0, 8*sizeof(int) );
			memset( binIndexPtr+2*(i*width+width-4), 0, 8*sizeof(uint16_t) );
		}

		magnitudePtr[(i*width+width-2)*2] = 0;
		magnitudePtr[(i*width+width-2)*2 + 1] = 0;
		magnitudePtr[(i*width+width-2)*2 + 2] = 0;
		magnitudePtr[(i*width+width-2)*2 + 3] = 0;

		binIndexPtr[(i*width+width-2)*2] = 0;
		binIndexPtr[(i*width+width-2)*2 + 1] = 0;
		binIndexPtr[(i*width+width-2)*2 + 2] = 0;
		binIndexPtr[(i*width+width-2)*2 + 3] = 0;
	}

	//zero out last two rows
	memset( magnitudePtr+2*width*(height-2), 0, (width*4)*sizeof(int) );
	memset( binIndexPtr+2*width*(height-2), 0, (width*4)*sizeof(uint16_t) );

	// ================ CALCULATING HISTOGRAMS =================	

	int* dst = (int*) malloc(height*width*nbins*sizeof(int));

	// column-histogram values
	int* h_row = (int*) malloc(width*nbins*sizeof(int));

	// compute the first column-histograms

	memset( h_row, 0, width*nbins*sizeof(int) );
	
	for(int i = 0; i < ksize; i++ )
	{
		for ( int j = 0; j < width; j++ )
		{
			h_row[j*nbins + binIndexPtr[i*width*2+j*2]] += magnitudePtr[i*width*2+j*2];
			h_row[j*nbins + binIndexPtr[i*width*2+j*2+1]] += magnitudePtr[i*width*2+j*2+1];
		}
	}

	// set first columns to zero (border effect)
	for(int k=0;k<r;k++)
	{
		memset( dst + k*width*nbins, 0, width*nbins*sizeof(int) );
	}

	// FOR EACH ROW
	for(int i = 0;; i++ )
	{
		__m128i _H0 = _mm_setzero_si128();
		__m128i _H1 = _mm_setzero_si128();
		__m128i _H2 = _mm_setzero_si128();
		__m128i _H3 = _mm_setzero_si128();

		// compute first histogram
		for(int j = 0; j < ksize; j++ )
		{
			__m128i _x0 = _mm_load_si128((__m128i*)(h_row+j*nbins)), _x1 =  _mm_load_si128((__m128i*)(h_row+j*nbins+4));
			__m128i _x2 = _mm_load_si128((__m128i*)(h_row+j*nbins+8)), _x3 =  _mm_load_si128((__m128i*)(h_row+j*nbins+12));
			_H0 = _mm_add_epi32(_H0,_x0);
			_H1 = _mm_add_epi32(_H1,_x1);
			_H2 = _mm_add_epi32(_H2,_x2);
			_H3 = _mm_add_epi32(_H3,_x3);
		}

		// set left most columns to zero (border effect)
		memset( dst + i*width*nbins, 0, r*nbins*sizeof(int) );

		// store first histogram values
		_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins+r*nbins), _H0);
		_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins+r*nbins+4), _H1);
		_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins+r*nbins+8), _H2);
		_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins+r*nbins+12), _H3);

		// calculate and store the rest of the histograms along the row i
		for ( int j = 0; j < width-ksize; j++ )
		{
			__m128i _x0 = _mm_load_si128((__m128i*)(h_row+j*nbins)), _x1 = _mm_load_si128((__m128i*)(h_row+j*nbins+4));
			__m128i _x2 = _mm_load_si128((__m128i*)(h_row+j*nbins+8)), _x3 = _mm_load_si128((__m128i*)(h_row+j*nbins+12));
			__m128i _y0 = _mm_load_si128((__m128i*)(h_row+(j+ksize)*nbins)), _y1 = _mm_load_si128((__m128i*)(h_row+(j+ksize)*nbins+4));
			__m128i _y2 = _mm_load_si128((__m128i*)(h_row+(j+ksize)*nbins+8)), _y3 = _mm_load_si128((__m128i*)(h_row+(j+ksize)*nbins+12));

			_H0 = _mm_sub_epi32(_H0,_x0);
			_H1 = _mm_sub_epi32(_H1,_x1);
			_H2 = _mm_sub_epi32(_H2,_x2);
			_H3 = _mm_sub_epi32(_H3,_x3);

			_H0 = _mm_add_epi32(_H0,_y0);
			_H1 = _mm_add_epi32(_H1,_y1);
			_H2 = _mm_add_epi32(_H2,_y2);
			_H3 = _mm_add_epi32(_H3,_y3);

			_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins + (j+1+r)*nbins), _H0);
			_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins + (j+1+r)*nbins + 4), _H1);
			_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins + (j+1+r)*nbins + 8), _H2);
			_mm_storeu_si128((__m128i*)(dst + (i+r)*width*nbins + (j+1+r)*nbins + 12), _H3);
		}

		// set right most columns to zero (border effect)
		memset( dst + (i*width+width-r)*nbins, 0, r*nbins*sizeof(int) );

		// check if time to stop
		if( i+ksize >= height )
			break;

		// update column-histograms 
		for ( int j = 0; j < width; j++ )
		{
			h_row[j*nbins + binIndexPtr[i*width*2+j*2]] -= magnitudePtr[i*width*2+j*2];
			h_row[j*nbins + binIndexPtr[i*width*2+j*2+1]] -= magnitudePtr[i*width*2+j*2+1];
	
			h_row[j*nbins + binIndexPtr[(i+ksize)*width*2+j*2]] += magnitudePtr[(i+ksize)*width*2+j*2];
			h_row[j*nbins + binIndexPtr[(i+ksize)*width*2+j*2+1]] += magnitudePtr[(i+ksize)*width*2+j*2+1];
		}
	}
	
	// set last r rows to zero (border effect)
	memset( dst + (height-(r+1))*width*nbins, 0, (r+1)*width*nbins*sizeof(int) );

	// ================ MAX VALUES =============================

	float maxVals[16];

	__m128i max0 = _mm_setzero_si128();
	__m128i max1 = _mm_setzero_si128();
	__m128i max2 = _mm_setzero_si128();
	__m128i max3 = _mm_setzero_si128();
	
	for(int i=0;i<width*height;i++)
	{
		__m128i x0 = _mm_load_si128((__m128i*)(dst + i*nbins));
		__m128i x1 = _mm_load_si128((__m128i*)(dst + i*nbins + 4));
		__m128i x2 = _mm_load_si128((__m128i*)(dst + i*nbins + 8));
		__m128i x3 = _mm_load_si128((__m128i*)(dst + i*nbins + 12));

		max0 = _mm_max_epi32(x0,max0);
		max1 = _mm_max_epi32(x1,max1);
		max2 = _mm_max_epi32(x2,max2);
		max3 = _mm_max_epi32(x3,max3);
	}
	
	__m128 values0 = _mm_cvtepi32_ps(max0);
	__m128 values1 = _mm_cvtepi32_ps(max1);
	__m128 values2 = _mm_cvtepi32_ps(max2);
	__m128 values3 = _mm_cvtepi32_ps(max3);

	_mm_store_ps(maxVals, values0);
	_mm_store_ps((maxVals+4), values1);
	_mm_store_ps(maxVals+8, values2);
	_mm_store_ps((maxVals+12), values3);

	for(int k=0;k<16;k++) maxVals[k] = maxVals[k] == 0 ? 1.0f : 255.0f/maxVals[k];

	__m128 s0 = _mm_load_ps(maxVals);
	__m128 s1 = _mm_load_ps(maxVals+4);
	__m128 s2 = _mm_load_ps(maxVals+8);
	__m128 s3 = _mm_load_ps(maxVals+12);

	// ================ NORMALIZATION IN FINAL SAVING =============================

	__m128i zeros = _mm_setzero_si128();

	for(int i=0;i<=width*height-8;i+=8)
	{
		__m128 x0 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins))); // 0,1,2,3
		__m128 x1 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 4))); // 4,5,6,7
		__m128 x2 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 8))); // 8,9,10,11
		__m128 x3 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 12))); // 12,13,14,15

		__m128 x4 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 16))); // 0,1,2,3
		__m128 x5 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 20))); // 4,5,6,7
		__m128 x6 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 24))); // 8,9,10,11
		__m128 x7 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 28))); // 12,13,14,15

		__m128 x8 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 32))); // 0,1,2,3
		__m128 x9 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 36))); // 4,5,6,7
		__m128 x10 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 40))); // 8,9,10,11
		__m128 x11 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 44))); // 12,13,14,15

		__m128 x12 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 48))); // 0,1,2,3
		__m128 x13 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 52))); // 4,5,6,7
		__m128 x14 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 56))); // 8,9,10,11
		__m128 x15 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 60))); // 12,13,14,15

		__m128 _x0 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 64))); // 0,1,2,3
		__m128 _x1 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 68))); // 4,5,6,7
		__m128 _x2 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 72))); // 8,9,10,11
		__m128 _x3 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 76))); // 12,13,14,15

		__m128 _x4 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 80))); // 0,1,2,3
		__m128 _x5 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 84))); // 4,5,6,7
		__m128 _x6 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 88))); // 8,9,10,11
		__m128 _x7 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 92))); // 12,13,14,15

		__m128 _x8 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 96))); // 0,1,2,3
		__m128 _x9 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 100))); // 4,5,6,7
		__m128 _x10 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 104))); // 8,9,10,11
		__m128 _x11 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 108))); // 12,13,14,15

		__m128 _x12 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 112))); // 0,1,2,3
		__m128 _x13 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 116))); // 4,5,6,7
		__m128 _x14 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 120))); // 8,9,10,11
		__m128 _x15 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(dst + i*nbins + 124))); // 12,13,14,15

		x0 = _mm_mul_ps(x0,s0);
		x1 = _mm_mul_ps(x1,s1);
		x2 = _mm_mul_ps(x2,s2);
		x3 = _mm_mul_ps(x3,s3);
		x4 = _mm_mul_ps(x4,s0);
		x5 = _mm_mul_ps(x5,s1);
		x6 = _mm_mul_ps(x6,s2);
		x7 = _mm_mul_ps(x7,s3);
		x8 = _mm_mul_ps(x8,s0);
		x9 = _mm_mul_ps(x9,s1);
		x10 = _mm_mul_ps(x10,s2);
		x11 = _mm_mul_ps(x11,s3);
		x12 = _mm_mul_ps(x12,s0);
		x13 = _mm_mul_ps(x13,s1);
		x14 = _mm_mul_ps(x14,s2);
		x15 = _mm_mul_ps(x15,s3);
		_x0 = _mm_mul_ps(_x0,s0);
		_x1 = _mm_mul_ps(_x1,s1);
		_x2 = _mm_mul_ps(_x2,s2);
		_x3 = _mm_mul_ps(_x3,s3);
		_x4 = _mm_mul_ps(_x4,s0);
		_x5 = _mm_mul_ps(_x5,s1);
		_x6 = _mm_mul_ps(_x6,s2);
		_x7 = _mm_mul_ps(_x7,s3);
		_x8 = _mm_mul_ps(_x8,s0);
		_x9 = _mm_mul_ps(_x9,s1);
		_x10 = _mm_mul_ps(_x10,s2);
		_x11 = _mm_mul_ps(_x11,s3);
		_x12 = _mm_mul_ps(_x12,s0);
		_x13 = _mm_mul_ps(_x13,s1);
		_x14 = _mm_mul_ps(_x14,s2);
		_x15 = _mm_mul_ps(_x15,s3);


		// 0,1,0,1
		__m128 ft04l = _mm_unpacklo_ps(x0, x4);
		__m128 ft812l = _mm_unpacklo_ps(x8, x12);
		// 2,3,2,3
		__m128 ft04h = _mm_unpackhi_ps(x0, x4);
		__m128 ft812h = _mm_unpackhi_ps(x8, x12);
		// 4,5,4,5
		__m128 ft15l = _mm_unpacklo_ps(x1, x5);
		__m128 ft913l = _mm_unpacklo_ps(x9, x13);
		// 6,7,6,7
		__m128 ft15h = _mm_unpackhi_ps(x1, x5);
		__m128 ft913h = _mm_unpackhi_ps(x9, x13);
		// 8,9,8,9
		__m128 ft26l = _mm_unpacklo_ps(x2, x6);
		__m128 ft1014l = _mm_unpacklo_ps(x10, x14);
		// 10,11,10,11
		__m128 ft26h = _mm_unpackhi_ps(x2, x6);
		__m128 ft1014h = _mm_unpackhi_ps(x10, x14);
		// 12,13,12,13
		__m128 ft37l = _mm_unpacklo_ps(x3, x7);
		__m128 ft1115l = _mm_unpacklo_ps(x11, x15);
		// 14,15,14,15
		__m128 ft37h = _mm_unpackhi_ps(x3, x7);
		__m128 ft1115h = _mm_unpackhi_ps(x11, x15);


		// 0,1,0,1
		__m128 _ft04l = _mm_unpacklo_ps(_x0, _x4);
		__m128 _ft812l = _mm_unpacklo_ps(_x8, _x12);
		// 2,3,2,3
		__m128 _ft04h = _mm_unpackhi_ps(_x0, _x4);
		__m128 _ft812h = _mm_unpackhi_ps(_x8, _x12);
		// 4,5,4,5
		__m128 _ft15l = _mm_unpacklo_ps(_x1, _x5);
		__m128 _ft913l = _mm_unpacklo_ps(_x9, _x13);
		// 6,7,6,7
		__m128 _ft15h = _mm_unpackhi_ps(_x1, _x5);
		__m128 _ft913h = _mm_unpackhi_ps(_x9, _x13);
		// 8,9,8,9
		__m128 _ft26l = _mm_unpacklo_ps(_x2, _x6);
		__m128 _ft1014l = _mm_unpacklo_ps(_x10, _x14);
		// 10,11,10,11
		__m128 _ft26h = _mm_unpackhi_ps(_x2, _x6);
		__m128 _ft1014h = _mm_unpackhi_ps(_x10, _x14);
		// 12,13,12,13
		__m128 _ft37l = _mm_unpacklo_ps(_x3, _x7);
		__m128 _ft1115l = _mm_unpacklo_ps(_x11, _x15);
		// 14,15,14,15
		__m128 _ft37h = _mm_unpackhi_ps(_x3, _x7);
		__m128 _ft1115h = _mm_unpackhi_ps(_x11, _x15);


		// 0,0,0,0
		x0 = _mm_shuffle_ps(ft04l, ft812l, _MM_SHUFFLE(1,0,1,0));
		// 1,1,1,1
		x1 = _mm_shuffle_ps(ft04l, ft812l, _MM_SHUFFLE(3,2,3,2));
		// 2,2,2,2
		x2 = _mm_shuffle_ps(ft04h, ft812h, _MM_SHUFFLE(1,0,1,0));
		// 3,3,3,3
		x3 = _mm_shuffle_ps(ft04h, ft812h, _MM_SHUFFLE(3,2,3,2));
		// 4,4,4,4
		x4 = _mm_shuffle_ps(ft15l, ft913l, _MM_SHUFFLE(1,0,1,0));
		// 5,5,5,5
		x5 = _mm_shuffle_ps(ft15l, ft913l, _MM_SHUFFLE(3,2,3,2));
		// 6,6,6,6
		x6 = _mm_shuffle_ps(ft15h, ft913h, _MM_SHUFFLE(1,0,1,0));
		// 7,7,7,7
		x7 = _mm_shuffle_ps(ft15h, ft913h, _MM_SHUFFLE(3,2,3,2));
		// 8,8,8,8
		x8 = _mm_shuffle_ps(ft26l, ft1014l, _MM_SHUFFLE(1,0,1,0));
		// 9,9,9,9
		x9 = _mm_shuffle_ps(ft26l, ft1014l, _MM_SHUFFLE(3,2,3,2));
		// 10,10,10,10
		x10 = _mm_shuffle_ps(ft26h, ft1014h, _MM_SHUFFLE(1,0,1,0));
		// 11,11,11,11
		x11 = _mm_shuffle_ps(ft26h, ft1014h, _MM_SHUFFLE(3,2,3,2));
		// 12,12,12,12
		x12 = _mm_shuffle_ps(ft37l, ft1115l, _MM_SHUFFLE(1,0,1,0));
		// 13,13,13,13
		x13 = _mm_shuffle_ps(ft37l, ft1115l, _MM_SHUFFLE(3,2,3,2));
		// 14,14,14,14
		x14 = _mm_shuffle_ps(ft37h, ft1115h, _MM_SHUFFLE(1,0,1,0));
		// 15,15,15,15
		x15 = _mm_shuffle_ps(ft37h, ft1115h, _MM_SHUFFLE(3,2,3,2));


		// 0,0,0,0
		_x0 = _mm_shuffle_ps(_ft04l, _ft812l, _MM_SHUFFLE(1,0,1,0));
		// 1,1,1,1
		_x1 = _mm_shuffle_ps(_ft04l, _ft812l, _MM_SHUFFLE(3,2,3,2));
		// 2,2,2,2
		_x2 = _mm_shuffle_ps(_ft04h, _ft812h, _MM_SHUFFLE(1,0,1,0));
		// 3,3,3,3
		_x3 = _mm_shuffle_ps(_ft04h, _ft812h, _MM_SHUFFLE(3,2,3,2));
		// 4,4,4,4
		_x4 = _mm_shuffle_ps(_ft15l, _ft913l, _MM_SHUFFLE(1,0,1,0));
		// 5,5,5,5
		_x5 = _mm_shuffle_ps(_ft15l, _ft913l, _MM_SHUFFLE(3,2,3,2));
		// 6,6,6,6
		_x6 = _mm_shuffle_ps(_ft15h, _ft913h, _MM_SHUFFLE(1,0,1,0));
		// 7,7,7,7
		_x7 = _mm_shuffle_ps(_ft15h, _ft913h, _MM_SHUFFLE(3,2,3,2));
		// 8,8,8,8
		_x8 = _mm_shuffle_ps(_ft26l, _ft1014l, _MM_SHUFFLE(1,0,1,0));
		// 9,9,9,9
		_x9 = _mm_shuffle_ps(_ft26l, _ft1014l, _MM_SHUFFLE(3,2,3,2));
		// 10,10,10,10
		_x10 = _mm_shuffle_ps(_ft26h, _ft1014h, _MM_SHUFFLE(1,0,1,0));
		// 11,11,11,11
		_x11 = _mm_shuffle_ps(_ft26h, _ft1014h, _MM_SHUFFLE(3,2,3,2));
		// 12,12,12,12
		_x12 = _mm_shuffle_ps(_ft37l, _ft1115l, _MM_SHUFFLE(1,0,1,0));
		// 13,13,13,13
		_x13 = _mm_shuffle_ps(_ft37l, _ft1115l, _MM_SHUFFLE(3,2,3,2));
		// 14,14,14,14
		_x14 = _mm_shuffle_ps(_ft37h, _ft1115h, _MM_SHUFFLE(1,0,1,0));
		// 15,15,15,15
		_x15 = _mm_shuffle_ps(_ft37h, _ft1115h, _MM_SHUFFLE(3,2,3,2));


		// 0s
		__m128i values = _mm_cvttps_epi32(x0);
		__m128i _values = _mm_cvttps_epi32(_x0);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin0_ptr + i), values);

		// 1s
		values = _mm_cvttps_epi32(x1);
		_values = _mm_cvttps_epi32(_x1);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin1_ptr + i), values);

		// 2s
		values = _mm_cvttps_epi32(x2);
		_values = _mm_cvttps_epi32(_x2);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin2_ptr + i), values);

		// 3s
		values = _mm_cvttps_epi32(x3);
		_values = _mm_cvttps_epi32(_x3);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin3_ptr + i), values);

		// 4s
		values = _mm_cvttps_epi32(x4);
		_values = _mm_cvttps_epi32(_x4);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin4_ptr + i), values);

		// 5s
		values = _mm_cvttps_epi32(x5);
		_values = _mm_cvttps_epi32(_x5);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin5_ptr + i), values);

		// 6s
		values = _mm_cvttps_epi32(x6);
		_values = _mm_cvttps_epi32(_x6);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin6_ptr + i), values);

		// 7s
		values = _mm_cvttps_epi32(x7);
		_values = _mm_cvttps_epi32(_x7);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin7_ptr + i), values);

		// 8s
		values = _mm_cvttps_epi32(x8);
		_values = _mm_cvttps_epi32(_x8);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin8_ptr + i), values);

		// 9s
		values = _mm_cvttps_epi32(x9);
		_values = _mm_cvttps_epi32(_x9);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin9_ptr + i), values);

		// 10s
		values = _mm_cvttps_epi32(x10);
		_values = _mm_cvttps_epi32(_x10);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin10_ptr + i), values);

		// 11s
		values = _mm_cvttps_epi32(x11);
		_values = _mm_cvttps_epi32(_x11);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin11_ptr + i), values);

		// 12s
		values = _mm_cvttps_epi32(x12);
		_values = _mm_cvttps_epi32(_x12);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin12_ptr + i), values);

		// 13s
		values = _mm_cvttps_epi32(x13);
		_values = _mm_cvttps_epi32(_x13);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin13_ptr + i), values);

		// 14s
		values = _mm_cvttps_epi32(x14);
		_values = _mm_cvttps_epi32(_x14);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin14_ptr + i), values);

		// 15s
		values = _mm_cvttps_epi32(x15);
		_values = _mm_cvttps_epi32(_x15);
		values = _mm_packs_epi32(values, _values);
		values = _mm_packus_epi16(values, zeros);
		_mm_storel_epi64((__m128i*) (bin15_ptr + i), values);
	}
	
	free(h_row);
	free(dst);
	free(magnitudePtr);
        free(binIndexPtr);
}
