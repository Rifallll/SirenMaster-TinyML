#include <Arduino.h>
#include <driver/i2s.h>
#include "model.h"
#include "dsp.h"

// TensorFlow Lite Micro
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

// I2S INMP441 Microphone Pins (Sesuaikan dengan wiring Anda)
#define I2S_WS 15
#define I2S_SD 13
#define I2S_SCK 2

#define I2S_PORT I2S_NUM_0

// Alokasi RAM (Penting: Wajib menggunakan PSRAM jika ada)
constexpr int kTensorArenaSize = 190000; // 190 KB
uint8_t* tensor_arena = nullptr;
int16_t* audio_ring_buffer = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

const char* CATEGORIES[] = {"AMBULANCE", "FIRETRUCK", "NORMAL", "POLICE"};

void setup_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE_HZ,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };
    
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    Serial.println("I2S Microphone (INMP441) Initialized.");
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n\n--- AI SIREN DETECTOR (ESP32) ---");
    
    // Alokasi Memori dengan PSRAM
    if (psramFound()) {
        Serial.println("PSRAM Ditemukan! Mengalokasikan memori di PSRAM...");
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
        audio_ring_buffer = (int16_t*)heap_caps_malloc(AUDIO_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    } else {
        Serial.println("Peringatan: PSRAM tidak ditemukan! Mengalokasikan di Internal RAM (Rawan kehabisan RAM!)");
        tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
        audio_ring_buffer = (int16_t*)malloc(AUDIO_SAMPLES * sizeof(int16_t));
    }
    
    if (!tensor_arena || !audio_ring_buffer) {
        Serial.println("ERROR: Gagal mengalokasikan memori! Alat berhenti.");
        while(1);
    }
    
    // Kosongkan buffer audio
    memset(audio_ring_buffer, 0, AUDIO_SAMPLES * sizeof(int16_t));
    
    setup_i2s();
    
    // Load Model
    model = tflite::GetModel(siren_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Versi TFLite schema tidak cocok!");
        while(1);
    }
    
    // Siapkan OP Resolver (Fungsi matematika yang dipakai AI)
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    
    // Bangun Interpreter
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() gagal! (Mungkin kTensorArenaSize kurang besar)");
        while(1);
    }
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("Model AI Berhasil Dimuat dan Siap Bekerja!");
    Serial.println("----------------------------------------");
}

void loop() {
    // 1. Baca 1 detik audio baru (8000 sampel)
    const int chunk_size = 8000;
    int16_t temp_buffer[chunk_size];
    size_t bytes_read = 0;
    
    i2s_read(I2S_PORT, &temp_buffer, chunk_size * sizeof(int16_t), &bytes_read, portMAX_DELAY);
    int samples_read = bytes_read / sizeof(int16_t);
    
    if (samples_read == chunk_size) {
        // 2. Geser Audio Ring Buffer ke Kiri sejauh 1 detik
        memmove(audio_ring_buffer, audio_ring_buffer + chunk_size, (AUDIO_SAMPLES - chunk_size) * sizeof(int16_t));
        
        // 3. Masukkan 1 detik audio baru ke bagian paling kanan (akhir)
        memcpy(audio_ring_buffer + (AUDIO_SAMPLES - chunk_size), temp_buffer, chunk_size * sizeof(int16_t));
        
        // 4. Ekstrak Fitur (Mel-Spectrogram DSP)
        float input_scale = input->params.scale;
        int input_zero_point = input->params.zero_point;
        
        int8_t* model_input_buffer = input->data.int8;
        
        // Fungsi ini akan mengekstrak suara 4 detik menjadi gambar dan menyuntikkannya ke input TFLite
        extract_features(audio_ring_buffer, AUDIO_SAMPLES, model_input_buffer, input_scale, input_zero_point);
        
        // 5. Jalankan Prediksi AI
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("ERROR: Invoke() gagal");
            return;
        }
        
        // 6. Baca Hasil (Dequantize output INT8 kembali ke Persentase / Float)
        float input_scale_out = output->params.scale;
        int input_zero_point_out = output->params.zero_point;
        
        float highest_prob = 0.0f;
        int best_idx = -1;
        
        Serial.print("Hasil: ");
        for (int i = 0; i < NUM_CLASSES; i++) {
            int8_t quant_val = output->data.int8[i];
            float prob = (quant_val - input_zero_point_out) * input_scale_out;
            
            if (prob > highest_prob) {
                highest_prob = prob;
                best_idx = i;
            }
        }
        
        // Tampilkan hanya jika yakin (misal > 75%)
        if (highest_prob > 0.75f && best_idx != 2) { // 2 adalah NORMAL
            Serial.printf(">>> TERDETEKSI: %s (%.0f%%) <<<\n", CATEGORIES[best_idx], highest_prob * 100);
        } else {
            Serial.println("NORMAL / Sunyi");
        }
    }
}
