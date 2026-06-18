/*
 * esp32_siren_detector.ino
 * TinyML Emergency Vehicle Siren Detector — ESP32/ESP32-S3
 * Menggunakan TensorFlow Lite Micro (Chirale_TensorFlowLite)
 *
 * WIRING (sesuaikan pin untuk ESP32-S3 jika perlu):
 * MAX9814 OUT → GPIO34 | Motor → GPIO25 | LED R/G/B → 26/27/14
 * ST7789: SCK→18, MOSI→23, CS→5, DC→2, RST→4, BL→15
 * Push Button → GPIO33 (active LOW, internal pull-up)
 *
 * Libraries: Chirale_TensorFlowLite, Adafruit_ST7789, Adafruit_GFX
 */

#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <Arduino.h>
#include <SPI.h>
#include <driver/i2s.h>

// ── TensorFlow Lite Micro (Chirale v2.0) ─────────────────────
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Chirale_TensorFlowLite.h>

#include "buttonmanager.h"
#include "model.h"


// ── Pin Definitions ──────────────────────────────────────────
// INMP441 Mic (I2S)
#define I2S_WS_PIN 32
#define I2S_SCK_PIN 33
#define I2S_SD_PIN 35

// Vibration Motor (Ubah ke 13 jika dihubungkan ke GPIO13)
#define MOTOR_PIN 25

// LED RGB (Active LOW atau Active HIGH tergantung modul)
#define LED_R_PIN 26
#define LED_G_PIN 27
#define LED_B_PIN 21

// LCD ST7789 (SPI)
#define LCD_SCL_PIN 18 // SPI SCK
#define LCD_SDA_PIN 23 // SPI MOSI
#define LCD_RES_PIN 4  // Reset
#define LCD_DC_PIN 22  // Data/Command
#define LCD_CS_PIN 14  // Chip Select

// Others
#define PIN_BUTTON 12 // Tombol di D12 (RTC GPIO - Mendukung Deep Sleep Wakeup)
#define TFT_BL 15     // Backlight

// ── Audio (dari model.h defines) ─────────────────────────────
#define SAMPLE_RATE SAMPLE_RATE_HZ
#define AUDIO_LEN AUDIO_SAMPLES
#define FFT_N N_FFT_SIZE
#define FFT_BINS N_FFT_BINS
#define HOP_LEN N_HOP_LENGTH
#define N_FRAMES N_TIME_FRAMES

#define N_MELS N_MEL_FILTERS
#define SAMPLE_US 125

// ── Microphone Calibration ───────────────────────────────────
#define MIC_GAIN 2.5f  // Pengali volume suara mic agar setara dengan laptop

// ── Inference ────────────────────────────────────────────────
#define CONFIDENCE_THR 0.92f
#define TENSOR_ARENA_KB 85
#define SMOOTH_COUNT 3

const char *CLASS_LABELS[NUM_CLASSES] = {"AMBULANCE", "FIRETRUCK", "NOISE",
                                         "POLICE"};

// ═══════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════
// Hardware SPI (lebih cepat & stabil daripada Software SPI)
Adafruit_ST7789 tft = Adafruit_ST7789(LCD_CS_PIN, LCD_DC_PIN, LCD_RES_PIN);

// TFLite Micro — MicroMutableOpResolver
static tflite::MicroMutableOpResolver<14> resolver;
static const tflite::Model *tflModel = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *inputTensor = nullptr;
static TfLiteTensor *outputTensor = nullptr;

constexpr int kTensorArenaSize = TENSOR_ARENA_KB * 1024;
static uint8_t *tensor_arena = nullptr;

// Audio & DSP buffers
static int16_t *chunkBuffer = nullptr;   // 1 detik (16KB)
static int16_t *historyBuffer = nullptr; // 4 detik (64KB)
static int historyHead = 0;              // index sampel tertua
SemaphoreHandle_t audioSemaphore;
TaskHandle_t TaskAudio;
TaskHandle_t TaskInference;
static float fftReal[FFT_N];
static float fftImag[FFT_N];
static float melEnergy[N_MELS];
static float outputScores[NUM_CLASSES];
static float powerSpec[FFT_BINS];

// Motor
static unsigned long motorTimer = 0;
static int motorState = 0, motorPattern = 0;

// Button
ButtonManager btnManager(PIN_BUTTON);
static SemaphoreHandle_t lcdMutex = NULL;

// Smart detect
static int lastDetectedClass = 2, consecutiveCount = 0, confirmedClass = 2;

// Visualizer and Strobe globals
static float visualizerPhase = 0;
static float targetAmplitude = 0;
static float currentAmplitude = 0;
static unsigned long lastVisualizerUpdate = 0;
static unsigned long lastStrobeUpdate = 0;
static bool strobeToggle = false;

// ═══════════════════════════════════════════════════════════════
// AUDIO
// ═══════════════════════════════════════════════════════════════
void audioTaskCode(void *pvParameters) {
  const int chunk_size = SAMPLE_RATE; // 1 detik = 8000 sampel
  size_t bytesIn = 0;

  // Membaca data dalam blok besar untuk menghemat CPU (Sangat penting agar
  // FreeRTOS tidak lag!)
  const int raw_chunk_size = 512; // 512 sampel 16kHz = 32ms audio
  int32_t *raw_buf = (int32_t *)malloc(raw_chunk_size * sizeof(int32_t));

  int chunk_idx = 0;

  for (;;) {
    if (!btnManager.isActive()) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    // 1. Baca 1 detik audio (16000 sampel fisik) lalu downsample ke 8000 sampel
    // Serta kalibrasikan volume dengan pengali gain 2.5x agar setara volume
    // laptop
    while (chunk_idx < chunk_size) {
      esp_err_t result = i2s_read(I2S_NUM_0, raw_buf, raw_chunk_size * 4,
                                  &bytesIn, pdMS_TO_TICKS(100));
      int samples_read = bytesIn / 4;

      if (result == ESP_OK && samples_read > 0) {
        // Lakukan downsampling (decimation by 2) dan gain boost 2.5x dalam
        // memori
        for (int j = 0; j < samples_read - 1; j += 2) {
          if (chunk_idx >= chunk_size)
            break;

          int16_t s1 = (int16_t)(raw_buf[j] >> 16);
          int16_t s2 = (int16_t)(raw_buf[j + 1] >> 16);
          float boosted = (((float)s1 + s2) / 2.0f) * MIC_GAIN;
          chunkBuffer[chunk_idx++] =
              (int16_t)constrain(boosted, -32768.0f, 32767.0f);
        }
      } else {
        // Jika gagal, isi sisa chunk dengan 0
        while (chunk_idx < chunk_size) {
          chunkBuffer[chunk_idx++] = 0;
        }
        Serial.println("[ERROR] i2s_read Timeout / Mic Error!");
      }
    }

    // Reset index untuk detik berikutnya
    chunk_idx = 0;

    // 2. Salin chunkBuffer ke historyBuffer (Ring Buffer)
    for (int i = 0; i < chunk_size; i++) {
      historyBuffer[historyHead] = chunkBuffer[i];
      historyHead = (historyHead + 1) % AUDIO_LEN;
    }

    // 3. Beri tahu Inference Task untuk memproses audio
    xSemaphoreGive(audioSemaphore);
  }
}

// ═══════════════════════════════════════════════════════════════
// FFT & MFCC
// ═══════════════════════════════════════════════════════════════
void computeFFT(float *vR, float *vI, int n) {
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = vR[i];
      vR[i] = vR[j];
      vR[j] = tr;
      float ti = vI[i];
      vI[i] = vI[j];
      vI[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * PI / (float)len;
    float wRe = cosf(ang), wIm = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cuRe = 1.0f, cuIm = 0.0f;
      for (int k = 0; k < (len >> 1); k++) {
        float uRe = vR[i + k], uIm = vI[i + k];
        float vRe =
            vR[i + k + (len >> 1)] * cuRe - vI[i + k + (len >> 1)] * cuIm;
        float vIm =
            vR[i + k + (len >> 1)] * cuIm + vI[i + k + (len >> 1)] * cuRe;
        vR[i + k] = uRe + vRe;
        vI[i + k] = uIm + vIm;
        vR[i + k + (len >> 1)] = uRe - vRe;
        vI[i + k + (len >> 1)] = uIm - vIm;
        float nr = cuRe * wRe - cuIm * wIm;
        cuIm = cuRe * wIm + cuIm * wRe;
        cuRe = nr;
      }
    }
  }
}

void extractMelSpec(int active_chunks) {
  // Auto-detect input tensor type (Dynamic Range model = float32, Full INT8 = int8)
  bool isFloat32Input = (inputTensor->type == kTfLiteFloat32);
  float *df = isFloat32Input ? inputTensor->data.f : nullptr;
  int8_t *di = isFloat32Input ? nullptr : inputTensor->data.int8;
  float sc = isFloat32Input ? 1.0f : inputTensor->params.scale;
  int   zp = isFloat32Input ? 0    : inputTensor->params.zero_point;

  int active_samples = active_chunks * SAMPLE_RATE;
  if (active_samples > AUDIO_LEN) active_samples = AUDIO_LEN;
  if (active_samples < SAMPLE_RATE) active_samples = SAMPLE_RATE;

  // --- PEAK NORMALIZATION / AUTO-GAIN (Max 10x Boost) ---
  float max_val = 0.0f;
  for (int idx = 0; idx < active_samples; idx++) {
    int tiled_chrono_idx = (AUDIO_LEN - active_samples) + (idx % active_samples);
    int curr = (historyHead + tiled_chrono_idx) % AUDIO_LEN;
    float val = fabsf((float)historyBuffer[curr]);
    if (val > max_val) {
      max_val = val;
    }
  }
  float gain = 1.0f;
  if (max_val > 1.0f) {
    gain = 32767.0f / max_val;
    if (gain > 10.0f) {
      gain = 10.0f;
    }
  }

  for (int frame = 0; frame < N_FRAMES; frame++) {
    int start = frame * HOP_LEN;

    // 1. Hitung rata-rata (mean) untuk DC offset removal
    float sum = 0.0f;
    for (int k = 0; k < FFT_N; k++) {
      int idx = start + k;
      int tiled_chrono_idx = (AUDIO_LEN - active_samples) + (idx % active_samples);
      int curr = (historyHead + tiled_chrono_idx) % AUDIO_LEN;
      int prev = (curr == 0) ? (AUDIO_LEN - 1) : (curr - 1);
      int next = (curr + 1) % AUDIO_LEN;

      // Low Pass Filter sederhana + Peak-Normalization Gain
      float sample =
          ((float)historyBuffer[prev] + historyBuffer[curr] + historyBuffer[next]) /
          (3.0f * 32768.0f) * gain;
      fftReal[k] = sample;
      sum += sample;
    }
    float mean = sum / FFT_N;

    // 2. DC removal & Hamming windowing
    for (int k = 0; k < FFT_N; k++) {
      fftReal[k] = (fftReal[k] - mean) * pgm_read_float(&HAMMING_WINDOW[k]);
      fftImag[k] = 0.0f;
    }

    computeFFT(fftReal, fftImag, FFT_N);
    for (int k = 0; k < FFT_BINS; k++)
      powerSpec[k] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k];

    for (int m = 0; m < N_MELS; m++) {
      float energy = 0.0f;
      for (int k = 0; k < FFT_BINS; k++)
        energy += pgm_read_float(&MEL_FILTERBANK[m][k]) * powerSpec[k];
      float log_mel = logf(energy + 1e-9f);

      float mean_m = pgm_read_float(&MEL_MEAN[m]);
      float std_m  = pgm_read_float(&MEL_STD[m]);
      float feat   = (log_mel - mean_m) / std_m;

      int tensor_idx = frame * N_MELS + m;
      if (isFloat32Input) {
        // Dynamic Range model: tulis float32 langsung
        df[tensor_idx] = feat;
      } else {
        // Full INT8 model: quantize dulu
        int q = (int)roundf(feat / sc) + zp;
        di[tensor_idx] = (int8_t)constrain(q, -128, 127);
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// INFERENCE
// ═══════════════════════════════════════════════════════════════
int runInference(float &confidence) {
  if (!interpreter) {
    confidence = 0;
    return 2;
  }
  // Fitur sudah dikuantisasi dan berada di inputTensor->data.int8 lewat
  // extractMelSpec()

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] Invoke gagal!");
    confidence = 0;
    return 2;
  }

  if (outputTensor->type == kTfLiteInt8) {
    float sc = outputTensor->params.scale;
    int zp = outputTensor->params.zero_point;
    for (int i = 0; i < NUM_CLASSES; i++)
      outputScores[i] = ((float)outputTensor->data.int8[i] - zp) * sc;
  } else {
    for (int i = 0; i < NUM_CLASSES; i++)
      outputScores[i] = outputTensor->data.f[i];
  }

  int best = 0;
  float bestS = outputScores[0];
  for (int i = 1; i < NUM_CLASSES; i++)
    if (outputScores[i] > bestS) {
      bestS = outputScores[i];
      best = i;
    }
  confidence = bestS;

  return (bestS >= CONFIDENCE_THR) ? best : 2;
}

// ── Variabel EMA State & Adaptive Noise Floor ──
static float ema_probs[NUM_CLASSES] = {0.0f, 0.0f, 1.0f, 0.0f};
#define EMA_ALPHA 0.15f
#define SIREN_THRESHOLD 0.80f
#define OVERRIDE_THRESHOLD 0.95f
static int locked_class_idx = 2; // Default NORMAL (2)

static float noise_floor = 400.0f; // Nilai awal lantai kebisingan
#define NOISE_FLOOR_MIN 200.0f
#define NOISE_FLOOR_MAX 1200.0f
#define NOISE_FLOOR_ALPHA 0.005f // Adaptasi sangat lambat (~20-30 detik)

int smartDetectEMA(float &out_prob, bool &is_thinking) {
  // 1. Clamp dan Normalisasi outputScores (dari Model)
  float clean_scores[NUM_CLASSES];
  float score_sum = 0.0f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    clean_scores[i] = outputScores[i] < 0.0f ? 0.0f : outputScores[i];
    score_sum += clean_scores[i];
  }
  if (score_sum > 0.0f) {
    for (int i = 0; i < NUM_CLASSES; i++) {
      clean_scores[i] /= score_sum;
    }
  } else {
    for (int i = 0; i < NUM_CLASSES; i++) {
      clean_scores[i] = (i == 2) ? 1.0f : 0.0f;
    }
  }

  // 2. Update EMA dengan clean_scores (MENGGUNAKAN DYNAMIC ADAPTIVE EMA)
  for (int i = 0; i < NUM_CLASSES; i++) {
    float current_alpha = EMA_ALPHA; // Default 0.15f (lambat & hati-hati)
    
    // [KECERDASAN BUATAN] Jika AI sangat yakin (>85%) pada frame ini (bukan Normal),
    // maka refleks ESP32 dipercepat 3x lipat agar langsung terdeteksi <1 detik!
    if (clean_scores[i] > 0.85f && i != 2) {
      current_alpha = 0.50f;
    }
    
    ema_probs[i] =
        (1.0f - current_alpha) * ema_probs[i] + current_alpha * clean_scores[i];
  }

  // 3. Re-normalisasi ema_probs
  float ema_sum = 0.0f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    ema_sum += ema_probs[i];
  }
  if (ema_sum > 0.0f) {
    for (int i = 0; i < NUM_CLASSES; i++) {
      ema_probs[i] /= ema_sum;
    }
  } else {
    for (int i = 0; i < NUM_CLASSES; i++) {
      ema_probs[i] = (i == 2) ? 1.0f : 0.0f;
    }
  }

  // 4. Cari Best Probability
  int best_idx = 0;
  float best_prob = ema_probs[0];
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (ema_probs[i] > best_prob) {
      best_prob = ema_probs[i];
      best_idx = i;
    }
  }

  is_thinking = false;

  // 5. Logika Lock-on
  if (locked_class_idx != 2) {
    if (ema_probs[locked_class_idx] < SIREN_THRESHOLD) {
      locked_class_idx = 2; // Lepas lock
    }
  }

  if (locked_class_idx == 2) {
    if (best_idx != 2 && best_prob >= SIREN_THRESHOLD) {
      locked_class_idx = best_idx;
      out_prob = best_prob;
    } else if (best_idx != 2 && best_prob > 0.40f) {
      // THINKING STATE!
      is_thinking = true;
      out_prob = best_prob;
      return best_idx; // Kembalikan kelas yang sedang dinilai
    } else {
      out_prob = ema_probs[2]; // NORMAL
      return 2;
    }
  } else {
    // Override logic
    if (best_idx != 2 && best_idx != locked_class_idx &&
        best_prob >= OVERRIDE_THRESHOLD) {
      locked_class_idx = best_idx;
      for (int i = 0; i < NUM_CLASSES; i++)
        ema_probs[i] = 0.0f;
      ema_probs[best_idx] = 1.0f;
      out_prob = best_prob;
    } else {
      out_prob = ema_probs[locked_class_idx];
    }
  }

  return locked_class_idx;
}

static int lastShownClass = -1;
static float lastShownConf = -1.0f;

// Helper function to draw centered text on the TFT display
void drawCenteredText(const char *text, int y, int sz, uint16_t color) {
  tft.setTextSize(sz);
  tft.setTextColor(color);
  int len = strlen(text);
  int x = (240 - (len * 6 * sz)) / 2;
  if (x < 0) x = 0;
  tft.setCursor(x, y);
  tft.print(text);
}

void lcdShowDetection(int cls, float conf, bool thinking) {
  if (lcdMutex == NULL ||
      xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(100)) != pdTRUE)
    return;

  // Modifikasi agar state thinking dianggap state yang berbeda secara UI
  int stateID = thinking ? (cls + 10) : cls;

  if (stateID != lastShownClass) {
    lastShownClass = stateID;
    lastShownConf = conf;
    strobeToggle = true; // Mulai dengan warna menyala
    lastStrobeUpdate = millis(); // Tunda strobo pertama agar warna menyala pertama terlihat penuh

    uint16_t bg = ST77XX_BLACK;
    uint16_t fg = ST77XX_WHITE;
    const char *alertText = "";

    if (thinking) {
      bg = 0x8200; // Merah Tua / Coklat Redup (Bukan merah menyala)
      fg = 0xFD20; // Oranye Kuning
      alertText = CLASS_LABELS[cls];
    } else {
      switch (cls) {
      case 0:
        bg = ST77XX_RED;
        fg = ST77XX_WHITE;
        alertText = "AMBULANCE";
        break;
      case 1:
        bg = 0xFD20; /* Orange */
        fg = ST77XX_BLACK;
        alertText = "FIRETRUCK";
        break;
      case 2:
        bg = ST77XX_BLACK;
        fg = ST77XX_GREEN;
        alertText = "SAFE";
        break;
      case 3:
        bg = ST77XX_BLUE;
        fg = ST77XX_WHITE;
        alertText = "POLICE";
        break;
      }
    }

    tft.fillScreen(bg);

    if (cls == 2 && !thinking) {
      // Tampilan LISTENING / SCANNING sederhana & bersih
      drawCenteredText("SIREN MASTER", 80, 2, ST77XX_GREEN);
      drawCenteredText("SAFE", 150, 4, ST77XX_GREEN);
      drawCenteredText("SCANNING...", 230, 2, ST77XX_GREEN);
    } else if (thinking) {
      // Tampilan MENILAI...
      drawCenteredText("MENILAI...", 70, 3, fg);
      drawCenteredText(alertText, 150, 3, fg);
      
      char confBuf[20];
      sprintf(confBuf, "Conf: %3.0f%%", conf * 100.0f);
      drawCenteredText(confBuf, 230, 2, fg);
    } else {
      // Tampilan ALERT Megah
      drawCenteredText("WARNING!", 70, 4, fg);
      drawCenteredText(alertText, 160, 4, fg);

      char confBuf[20];
      sprintf(confBuf, "Conf: %3.0f%%", conf * 100.0f);
      drawCenteredText(confBuf, 240, 2, fg);
    }
  } else if (cls != 2 || thinking) {
    // Jika state ID sama, tetapi probabilitas berubah, perbarui confidence secara real-time
    int oldPct = (int)roundf(lastShownConf * 100.0f);
    int newPct = (int)roundf(conf * 100.0f);
    if (oldPct != newPct) {
      lastShownConf = conf;
      if (thinking) {
        // Redraw baris confidence langsung pada background thinking (tidak strobo)
        uint16_t bg = 0x8200; // Merah Tua / Coklat Redup
        uint16_t fg = 0xFD20; // Oranye Kuning
        tft.fillRect(0, 230, 240, 16, bg);
        char confBuf[20];
        sprintf(confBuf, "Conf: %3.0f%%", conf * 100.0f);
        drawCenteredText(confBuf, 230, 2, fg);
      }
    }
  }
  xSemaphoreGive(lcdMutex);
}

void applyOutputs(int cls, float conf, bool thinking) {
  // Cegah AI menyalakan komponen jika sistem baru saja dimatikan
  if (!btnManager.isActive())
    return;

  if (cls == 2) {
    // SAFE: Matikan LED dan Motor
    setRGB(0, 0, 0);
    motorPattern = 0;
  } else if (thinking) {
    // Sedang MENILAI (Thinking): Nyalakan LED secara solid untuk respon instan, jangan getarkan motor dulu
    switch (cls) {
    case 0: setRGB(1, 0, 0); break; // Merah
    case 1: setRGB(1, 1, 0); break; // Kuning
    case 3: setRGB(0, 0, 1); break; // Biru
    }
    motorPattern = 0;
  } else {
    // ALERT PENUH: Nyalakan LED secara solid/strobo dan getarkan motor
    switch (cls) {
    case 0:
      setRGB(1, 0, 0);
      motorPattern = 1;
      break;
    case 1:
      setRGB(1, 1, 0);
      motorPattern = 2;
      break;
    case 3:
      setRGB(0, 0, 1);
      motorPattern = 3;
      break;
    }
  }

  lcdShowDetection(cls, conf, thinking);
}

// ═══════════════════════════════════════════════════════════════
// BUTTON
// ═══════════════════════════════════════════════════════════════
void checkButton() {
  if (btnManager.update()) {
    // Tombol ditekan -> MATIKAN ESP32 TOTAL (Deep Sleep)
    Serial.println("[POWER] Tombol ditekan! Mematikan semua komponen...");

    // 1. Matikan semua komponen output
    setRGB(0, 0, 0);
    motorPattern = 0;
    digitalWrite(MOTOR_PIN, LOW);

    // 2. Tampilkan animasi SHUTDOWN di layar
    if (lcdMutex != NULL) {
      xSemaphoreTake(lcdMutex, portMAX_DELAY);
    }

    // Tampilan Standby Sederhana
    tft.fillScreen(ST77XX_BLACK);
    drawCenteredText("SIREN MASTER", 80, 2, 0x4208);
    drawCenteredText("STANDBY", 160, 4, ST77XX_RED);
    drawCenteredText("[ Press to Wake ]", 240, 2, 0x4208);

    delay(1000); // Tampilkan standby screen selama 1 detik

    // 3. Matikan backlight layar
    digitalWrite(TFT_BL, LOW);

    if (lcdMutex != NULL) {
      xSemaphoreGive(lcdMutex);
    }

    // 4. MATIKAN ESP32 TOTAL via Deep Sleep
    btnManager.shutdown(PIN_BUTTON);
  }
}

// Menggambar gelombang live oscilloscope ketika mode SAFE
void updateLiveVisualizer() {
  // Dinonaktifkan agar layar hanya berisi teks sederhana tanpa grafik gelombang/oscilloscope
}

// Berkedip strobo ketika mendeteksi sirine
void updateAlertStrobe() {
  if (!btnManager.isActive())
    return;
  if (lastShownClass == 2 || lastShownClass == -1 || lastShownClass >= 10)
    return; // Jangan strobo jika SAFE atau THINKING

  unsigned long now = millis();
  if (now - lastStrobeUpdate < 200)
    return; // Strobo 5Hz (200ms sekali)

  if (lcdMutex == NULL || xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(20)) != pdTRUE)
    return;
  lastStrobeUpdate = now;

  strobeToggle = !strobeToggle;

  uint16_t bg = ST77XX_BLACK;
  uint16_t fg = ST77XX_WHITE;
  const char *alertText = "";

  switch (lastShownClass) {
  case 0:
    bg = strobeToggle ? ST77XX_RED : ST77XX_BLACK;
    fg = ST77XX_WHITE;
    alertText = "AMBULANCE";
    setRGB(strobeToggle ? 1 : 0, 0, 0); // Strobo LED Merah Sinkron
    break;
  case 1:
    bg = strobeToggle ? 0xFD20 /* Oranye */ : ST77XX_BLACK;
    fg = strobeToggle ? ST77XX_BLACK : ST77XX_WHITE;
    alertText = "FIRETRUCK";
    setRGB(strobeToggle ? 1 : 0, strobeToggle ? 1 : 0,
           0); // Strobo LED Kuning Sinkron
    break;
  case 3:
    bg = strobeToggle ? ST77XX_BLUE : ST77XX_BLACK;
    fg = ST77XX_WHITE;
    alertText = "POLICE";
    setRGB(0, 0, strobeToggle ? 1 : 0); // Strobo LED Biru Sinkron
    break;
  }

  tft.fillScreen(bg);

  drawCenteredText("WARNING!", 70, 4, fg);
  drawCenteredText(alertText, 160, 4, fg);

  char confBuf[20];
  sprintf(confBuf, "Conf: %3.0f%%", lastShownConf * 100.0f);
  drawCenteredText(confBuf, 240, 2, fg);

  xSemaphoreGive(lcdMutex);
}

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  digitalWrite(LED_R_PIN, r);
  digitalWrite(LED_G_PIN, g);
  digitalWrite(LED_B_PIN, b);
}

void updateMotor() {
  static int lastMotorPattern = 0;
  if (motorPattern == 0) {
    digitalWrite(MOTOR_PIN, LOW);
    lastMotorPattern = 0;
    motorState = 0;
    return;
  }
  unsigned long now = millis();
  
  // Jika pola motor baru saja diaktifkan, langsung nyalakan motor secara instan
  if (motorPattern != lastMotorPattern) {
    lastMotorPattern = motorPattern;
    motorState = 1;
    motorTimer = now;
    digitalWrite(MOTOR_PIN, HIGH);
    return;
  }

  if (motorPattern == 1) {
    if (now - motorTimer >= 100) {
      motorState ^= 1;
      motorTimer = now;
    }
  } else if (motorPattern == 2) {
    if (now - motorTimer >= (uint32_t)(motorState ? 300 : 200)) {
      motorState ^= 1;
      motorTimer = now;
    }
  } else if (motorPattern == 3) {
    static int step = 0;
    uint32_t iv[4] = {150, 150, 150, 600};
    if (now - motorTimer >= iv[step % 4]) {
      step = (step + 1) % 4;
      motorState = (step == 0 || step == 2) ? 1 : 0;
      motorTimer = now;
    }
  }
  digitalWrite(MOTOR_PIN, motorState);
}

void ioTaskCode(void *pvParameters) {
  for (;;) {
    checkButton();
    updateMotor();
    updateLiveVisualizer();
    updateAlertStrobe();
    vTaskDelay(pdMS_TO_TICKS(10)); // Polling 10ms
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== TinyML Siren Detector (TFLite Micro) ===");

  pinMode(LED_R_PIN, OUTPUT);
  pinMode(LED_G_PIN, OUTPUT);
  pinMode(LED_B_PIN, OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  btnManager.begin();
  digitalWrite(MOTOR_PIN, LOW);
  setRGB(0, 0, 0);

  // ── Setup I2S Mic INMP441 ──
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 16000, // Jalankan I2S fisik pada 16kHz agar modulasi clock
                            // INMP441 stabil & tidak terdistorsi
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
#if ESP_ARDUINO_VERSION_MAJOR >= 3
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#else
      .communication_format =
          (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
#endif
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 512,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};
  i2s_pin_config_t pin_config = {.bck_io_num = I2S_SCK_PIN,
                                 .ws_io_num = I2S_WS_PIN,
                                 .data_out_num = I2S_PIN_NO_CHANGE,
                                 .data_in_num = I2S_SD_PIN};
  esp_err_t err1 = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  esp_err_t err2 = i2s_set_pin(I2S_NUM_0, &pin_config);

  if (err1 != ESP_OK || err2 != ESP_OK) {
    Serial.printf("[ERROR] I2S Install Gagal! Err1: %d, Err2: %d\n", err1,
                  err2);
  } else {
    Serial.println("[OK] I2S Mic Berhasil Diinstall!");
  }

  // LCD
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
  delay(200);               // Beri waktu LCD untuk stabil setelah dinyalakan
  tft.setSPISpeed(8000000); // 8MHz (pelan tapi stabil untuk breadboard)
  tft.init(240, 320, SPI_MODE3); // Inisialisasi layar 240x320 (Full Screen!)
  tft.setRotation(2); // Putar layar agar tulisan tidak terbalik/miring

  // SPLASH SCREEN STARTUP MEWAH
  tft.fillScreen(ST77XX_BLACK);

  drawCenteredText("SIREN MASTER", 80, 2, ST77XX_CYAN);
  drawCenteredText("SYSTEM STARTING", 125, 1, ST77XX_WHITE);

  // Progress Bar Container
  tft.drawRect(55, 150, 130, 8, ST77XX_CYAN);
  tft.fillRect(57, 152, 15, 4, ST77XX_CYAN); // 10%
  drawCenteredText("Allocating RAM", 175, 1, ST77XX_WHITE);

  // ── ALOKASI MEMORI MURNI RAM INTERNAL (WROOM SAFE) ──
  tensor_arena = (uint8_t *)malloc(kTensorArenaSize);
  chunkBuffer = (int16_t *)malloc(SAMPLE_RATE * sizeof(int16_t));
  historyBuffer = (int16_t *)malloc(AUDIO_LEN * sizeof(int16_t));

  if (!tensor_arena || !chunkBuffer || !historyBuffer) {
    Serial.println("[ERROR] GAGAL Mengalokasikan Memori! Alat berhenti.");
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(30, 175);
    tft.println("OUT OF MEMORY!");
    while (1)
      delay(100);
  }
  memset(historyBuffer, 0, AUDIO_LEN * sizeof(int16_t));

  // 40% Progress
  tft.fillRect(57, 152, 50, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK); // clear text
  drawCenteredText("Loading AI Model", 175, 1, ST77XX_WHITE);

  // ── TFLite: Load model ──
  tflModel = tflite::GetModel(siren_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model version mismatch!");
    while (1)
      delay(1000);
  }

  // 60% Progress
  tft.fillRect(57, 152, 80, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK);
  drawCenteredText("Registering Kernels", 175, 1, ST77XX_WHITE);

  // ── Register HANYA ops yang dipakai model (hemat ~1MB RAM!) ──
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean(); // GlobalAveragePooling
  resolver.AddFullyConnected();
  resolver.AddRelu(); // Menangani aktivasi ReLU jika terpisah
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddExpandDims();

  // 80% Progress
  tft.fillRect(57, 152, 105, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK);
  drawCenteredText("Allocating Tensors", 175, 1, ST77XX_WHITE);

  // ── Buat interpreter ──
  static tflite::MicroInterpreter static_interpreter(
      tflModel, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("[ERROR] AllocateTensors gagal! Arena=%dKB\n",
                  TENSOR_ARENA_KB);
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(30, 175);
    tft.println("ALLOC ERROR!");
    while (1)
      delay(1000);
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  Serial.printf("[OK] Arena %dKB, used %d bytes\n", TENSOR_ARENA_KB,
                interpreter->arena_used_bytes());
  Serial.printf("[IN]  type=%d (%s) shape=(%d,%d,%d,%d)\n", inputTensor->type,
                inputTensor->type == kTfLiteFloat32 ? "FLOAT32" : "INT8",
                inputTensor->dims->data[0], inputTensor->dims->data[1],
                inputTensor->dims->data[2], inputTensor->dims->data[3]);
  Serial.printf("[OUT] type=%d (%s) shape=(%d,%d)\n", outputTensor->type,
                outputTensor->type == kTfLiteFloat32 ? "FLOAT32" : "INT8",
                outputTensor->dims->data[0], outputTensor->dims->data[1]);

  // 100% Progress
  tft.fillRect(57, 152, 126, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK);
  drawCenteredText("System Ready", 175, 1, ST77XX_WHITE);
  delay(300);

  // Startup test (LED & Haptic Pulse)
  setRGB(0, 1, 0); // Hijau
  digitalWrite(MOTOR_PIN, HIGH);

  tft.fillScreen(ST77XX_BLACK);

  tft.fillScreen(ST77XX_BLACK);

  drawCenteredText("SIREN MASTER", 80, 2, ST77XX_GREEN);
  drawCenteredText("READY", 150, 4, ST77XX_GREEN);
  drawCenteredText("SCANNING...", 230, 2, ST77XX_GREEN);

  delay(1000);
  digitalWrite(MOTOR_PIN, LOW);
  setRGB(0, 0, 0);
  delay(500);

  motorTimer = millis();

  // Start FreeRTOS Tasks
  audioSemaphore = xSemaphoreCreateBinary();
  lcdMutex = xSemaphoreCreateMutex();

  BaseType_t res1 =
      xTaskCreatePinnedToCore(audioTaskCode, "TaskAudio", 4096, NULL, 2,
                              &TaskAudio, 0); // Core 0 untuk I2S

  BaseType_t res2 =
      xTaskCreatePinnedToCore(inferenceTaskCode, "TaskInference", 8192, NULL, 1,
                              &TaskInference, 1); // Core 1 untuk AI

  BaseType_t res3 = xTaskCreatePinnedToCore(
      ioTaskCode, "TaskIO", 4096, NULL, 2, NULL,
      1); // Core 1, Priority 2 (Preempts AI) - Stack diperbesar ke 4096

  if (res1 != pdPASS || res2 != pdPASS || res3 != pdPASS) {
    Serial.println(
        "[FATAL ERROR] Gagal membuat FreeRTOS Task! Kehabisan Memori RAM.");
    tft.fillScreen(ST77XX_RED);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(10, 175);
    tft.println("TASK ERROR!");
    while (1)
      delay(100);
  }
}

const char* getShortLabel(int cls, bool thinking) {
  if (thinking) {
    switch (cls) {
      case 0: return "?AMBU";
      case 1: return "?FIRE";
      case 3: return "?POL";
      default: return "?SAFE";
    }
  } else {
    switch (cls) {
      case 0: return "AMBU";
      case 1: return "FIRE";
      case 2: return "SAFE";
      case 3: return "POL";
      default: return "SAFE";
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// LOOP
// ═══════════════════════════════════════════════════════════════
void inferenceTaskCode(void *pvParameters) {
  static int loud_chunks_count = 0;
  static bool is_silent_state = true;
  for (;;) {
    if (xSemaphoreTake(audioSemaphore, portMAX_DELAY) == pdTRUE) {
      if (!btnManager.isActive())
        continue;

      // --- NOISE GATE & CLIPPING DETECTOR ---
      float sumSq = 0;
      int clip_count = 0;
      for (int i = 0; i < AUDIO_LEN; i++) {
        float val = (float)historyBuffer[i];
        sumSq += (val * val);
        if (historyBuffer[i] >= 32760 || historyBuffer[i] <= -32760) {
          clip_count++;
        }
      }
      float rms = sqrt(sumSq / AUDIO_LEN);

      // Hitung threshold dinamis atas dan bawah dengan hysteresis
      float threshold_high = noise_floor + 150.0f;
      float threshold_low = noise_floor + 50.0f;
      
      bool check_silent = is_silent_state ? (rms < threshold_high) : (rms < threshold_low);

      if (check_silent) {
        is_silent_state = true;
        
        // Adaptasikan lantai kebisingan secara lambat saat hening/aman
        noise_floor = (1.0f - NOISE_FLOOR_ALPHA) * noise_floor + NOISE_FLOOR_ALPHA * rms;
        noise_floor = constrain(noise_floor, NOISE_FLOOR_MIN, NOISE_FLOOR_MAX);

        // (Debug dihilangkan agar Serial Monitor bersih saat testing)
        targetAmplitude = 0.0f; // Gelombang lurus

        // Reset EMA dan state lock secara instan saat hening (Noise Gate) agar
        // sesuai dengan logika Python
        ema_probs[0] = 0.0f;
        ema_probs[1] = 0.0f;
        ema_probs[2] = 1.0f;
        ema_probs[3] = 0.0f;
        locked_class_idx = 2;

        loud_chunks_count = 0; // Reset hitungan segment aktif saat hening

        bool thinking = false;
        applyOutputs(2, 1.0f, thinking);
        continue;
      }

      // Keluar dari hening
      is_silent_state = false;

      // Tambahkan hitungan segment suara aktif (maksimal 4 detik)
      loud_chunks_count = constrain(loud_chunks_count + 1, 1, 4);

      // Atur ketinggian gelombang visualizer berdasarkan volume suara
      targetAmplitude = constrain((rms - noise_floor) * 0.02f, 5.0f, 35.0f);

      // (Debug DSP dihilangkan)
      extractMelSpec(loud_chunks_count);

      // (Debug ML dihilangkan)
      float conf = 0;
      runInference(conf); // Updates outputScores[] internally

      bool thinking = false;
      float final_prob = 0.0f;
      int cls = smartDetectEMA(final_prob, thinking);
      applyOutputs(cls, final_prob, thinking);

      // Adaptasikan lantai kebisingan jika AI mengonfirmasi ini suara normal (SAFE)
      if (cls == 2 && !thinking) {
        noise_floor = (1.0f - NOISE_FLOOR_ALPHA) * noise_floor + NOISE_FLOOR_ALPHA * rms;
        noise_floor = constrain(noise_floor, NOISE_FLOOR_MIN, NOISE_FLOOR_MAX);
      }

      // Hitung kembali clean_scores untuk ditampilkan sebagai Raw Probs
      float clean_scores[NUM_CLASSES];
      float score_sum = 0.0f;
      for (int i = 0; i < NUM_CLASSES; i++) {
        clean_scores[i] = outputScores[i] < 0.0f ? 0.0f : outputScores[i];
        score_sum += clean_scores[i];
      }
      if (score_sum > 0.0f) {
        for (int i = 0; i < NUM_CLASSES; i++) {
          clean_scores[i] /= score_sum;
        }
      } else {
        for (int i = 0; i < NUM_CLASSES; i++) {
          clean_scores[i] = (i == 2) ? 1.0f : 0.0f;
        }
      }

      // (Raw & EMA debug dihilangkan agar Serial Monitor bersih)

      // Cetak output yang sama persis seperti terminal CMD Python
      const char* lbl = getShortLabel(cls, thinking);
      int bar_len = (int)(rms / 300.0f * 6.0f);
      if (bar_len < 0) bar_len = 0;
      if (bar_len > 6) bar_len = 6;
      
      String volBar = "";
      for (int b = 0; b < 6; b++) {
        if (b < bar_len) {
          volBar += "█";
        } else {
          volBar += "░";
        }
      }
      if (clip_count > 20) {
        volBar += " [!] CLIPPING (Kecilkan MIC_GAIN!)";
      }

      static int spinIdx = 0;
      const char spinner[] = {'|', '/', '-', '\\'};
      char spin = spinner[spinIdx % 4];
      spinIdx++;

      Serial.printf("[%c] %s (%.0f%%) | Vol: %s (%.0f)\n\n",
                    spin, lbl, final_prob * 100.0f, volBar.c_str(), rms);
    }
  }
}

void loop() {
  // Semua tugas sudah ditangani oleh FreeRTOS Tasks
  vTaskDelay(pdMS_TO_TICKS(1000));
}
