#include <Adafruit_SSD1351.h>

#ifndef DSP_H
#define DSP_H

#include <stdint.h>
#include <math.h>

// Deklarasi fungsi utama untuk mengekstrak fitur suara
// audio_buffer: Array dari suara mentah (16-bit PCM) sepanjang 32000 sampel (4 detik)
// output_features: Array kosong (ukuran 249 * 40) yang akan diisi dengan hasil Mel-Spectrogram
// input_scale & input_zero_point: Diambil dari model TFLite Micro di saat runtime
void extract_features(const int16_t* audio_buffer, int audio_len, int8_t* output_features, float input_scale, int input_zero_point);

#endif // DSP_H
