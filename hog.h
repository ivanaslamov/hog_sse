
#ifndef HOG_H
#define HOG_H

// simd optimized implementation that uses histogram addition rule
int histograms_of_gradient_directions_sse(unsigned char* src, unsigned char* bin0_ptr, unsigned char* bin1_ptr, unsigned char* bin2_ptr, unsigned char* bin3_ptr, unsigned char* bin4_ptr, unsigned char* bin5_ptr, unsigned char* bin6_ptr, unsigned char* bin7_ptr, unsigned char* bin8_ptr, unsigned char* bin9_ptr, unsigned char* bin10_ptr, unsigned char* bin11_ptr, unsigned char* bin12_ptr, unsigned char* bin13_ptr, unsigned char* bin14_ptr, unsigned char* bin15_ptr, int width, int height, int ksize );

#endif

