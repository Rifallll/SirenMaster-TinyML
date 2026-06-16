#include "dsp.h"
#include "model.h"
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fungsi pembantu untuk membalik bit (dibutuhkan oleh FFT)
static uint16_t reverse_bits(uint16_t val, int bits) {
    uint16_t res = 0;
    for (int i = 0; i < bits; i++) {
        res = (res << 1) | (val & 1);
        val >>= 1;
    }
    return res;
}

// Algoritma Radix-2 DIT FFT 
static void compute_fft(float* real, float* imag, int n) {
    int bits = 0;
    while ((1 << bits) < n) bits++;
    
    // Bit-reverse copy
    for (int i = 0; i < n; i++) {
        int rev = reverse_bits(i, bits);
        if (i < rev) {
            float temp_r = real[i];
            float temp_i = imag[i];
            real[i] = real[rev];
            imag[i] = imag[rev];
            real[rev] = temp_r;
            imag[rev] = temp_i;
        }
    }
    
    // Cooley-Tukey
    for (int step = 2; step <= n; step *= 2) {
        int half_step = step / 2;
        float angle = -2.0f * M_PI / step;
        float w_r = cosf(angle);
        float w_i = sinf(angle);
        
        for (int i = 0; i < n; i += step) {
            float curr_w_r = 1.0f;
            float curr_w_i = 0.0f;
            for (int j = 0; j < half_step; j++) {
                int a = i + j;
                int b = i + j + half_step;
                
                float t_r = curr_w_r * real[b] - curr_w_i * imag[b];
                float t_i = curr_w_r * imag[b] + curr_w_i * real[b];
                
                real[b] = real[a] - t_r;
                imag[b] = imag[a] - t_i;
                real[a] = real[a] + t_r;
                imag[a] = imag[a] + t_i;
                
                float next_w_r = curr_w_r * w_r - curr_w_i * w_i;
                float next_w_i = curr_w_r * w_i + curr_w_i * w_r;
                curr_w_r = next_w_r;
                curr_w_i = next_w_i;
            }
        }
    }
}

void extract_features(const int16_t* audio_buffer, int audio_len, int8_t* output_features, float input_scale, int input_zero_point) {
    // 1. Auto-Gain Normalization (Sama seperti Python)
    float max_val = 0.0f;
    for (int i = 0; i < audio_len; i++) {
        float val = fabsf((float)audio_buffer[i] / 32768.0f);
        if (val > max_val) max_val = val;
    }
    
    float gain = 1.0f;
    if (max_val > 1e-6f) {
        gain = 1.0f / max_val;
        if (gain > 10.0f) gain = 10.0f;
    }
    
    // 2. Sliding Window (STFT)
    float real[N_FFT_SIZE];
    float imag[N_FFT_SIZE];
    float power_spec[N_FFT_BINS];
    
    int out_idx = 0;
    
    for (int start = 0; start <= audio_len - N_FFT_SIZE && out_idx < N_TIME_FRAMES; start += N_HOP_LENGTH) {
        // Terapkan Hamming Window
        for (int i = 0; i < N_FFT_SIZE; i++) {
            float sample = ((float)audio_buffer[start + i] / 32768.0f) * gain;
            real[i] = sample * pgm_read_float(&HAMMING_WINDOW[i]);
            imag[i] = 0.0f;
        }
        
        // FFT
        compute_fft(real, imag, N_FFT_SIZE);
        
        // Power Spectrum (karena librosa menggunakan power=2.0 secara default)
        for (int i = 0; i < N_FFT_BINS; i++) {
            power_spec[i] = (real[i] * real[i] + imag[i] * imag[i]);
        }
        
        // Kalikan dengan Mel-Filterbank dan normalisasi
        for (int m = 0; m < N_MEL_FILTERS; m++) {
            float mel_energy = 0.0f;
            for (int k = 0; k < N_FFT_BINS; k++) {
                mel_energy += power_spec[k] * pgm_read_float(&MEL_FILTERBANK[m][k]);
            }
            
            // Logaritma natural
            float log_mel = logf(mel_energy + 1e-9f);
            
            // Z-Score Standardization
            float mean = pgm_read_float(&MEL_MEAN[m]);
            float std = pgm_read_float(&MEL_STD[m]);
            float feat_scaled = (log_mel - mean) / std;
            
            // Quantization ke INT8
            int quant_val = (int)roundf(feat_scaled / input_scale) + input_zero_point;
            if (quant_val > 127) quant_val = 127;
            if (quant_val < -128) quant_val = -128;
            
            // Simpan ke output array berformat [frame, mel_bin, 1]
            output_features[(out_idx * N_MEL_FILTERS) + m] = (int8_t)quant_val;
        }
        
        out_idx++;
    }
}
