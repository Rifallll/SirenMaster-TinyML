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

#include "model.h"
#include "buttonmanager.h"

// ── Pin Definitions ──────────────────────────────────────────
// INMP441 Mic (I2S)
#define I2S_WS_PIN   32
#define I2S_SCK_PIN  33
#define I2S_SD_PIN   35

// Vibration Motor
#define MOTOR_PIN    13

// LED RGB (Active LOW atau Active HIGH tergantung modul)
#define LED_R_PIN    26
#define LED_G_PIN    27
#define LED_B_PIN    21

// LCD ST7789 (SPI)
#define LCD_SCL_PIN  18   // SPI SCK
#define LCD_SDA_PIN  23   // SPI MOSI
#define LCD_RES_PIN  4    // Reset
#define LCD_DC_PIN   22   // Data/Command
#define LCD_CS_PIN   14   // Chip Select

// Others
#define PIN_BUTTON   12   // Tombol di D12 (RTC GPIO - Mendukung Deep Sleep Wakeup)
#define TFT_BL       15   // Backlight

// ── Audio (dari model.h defines) ─────────────────────────────
#define SAMPLE_RATE SAMPLE_RATE_HZ
#define AUDIO_LEN AUDIO_SAMPLES
#define FFT_N N_FFT_SIZE
#define FFT_BINS N_FFT_BINS
#define HOP_LEN N_HOP_LENGTH
#define N_FRAMES N_TIME_FRAMES

#define N_MELS N_MEL_FILTERS
#define SAMPLE_US 125

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
static uint8_t* tensor_arena = nullptr;

// Audio & DSP buffers
static int16_t* chunkBuffer = nullptr;   // 1 detik (16KB)
static int16_t* historyBuffer = nullptr; // 4 detik (64KB)
static int historyHead = 0; // index sampel tertua
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
  int32_t sample32 = 0;

  for(;;) {
    if (!btnManager.isActive()) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }
    
    // 1. Baca 1 detik audio dari I2S ke chunkBuffer
    for (int i = 0; i < chunk_size; i++) {
      esp_err_t result = i2s_read(I2S_NUM_0, &sample32, 4, &bytesIn, pdMS_TO_TICKS(100));
      if (result == ESP_OK && bytesIn == 4) {
        chunkBuffer[i] = (int16_t)(sample32 >> 16);
      } else {
        chunkBuffer[i] = 0;
        if (i == 0) {
           Serial.println("[ERROR] i2s_read Timeout / Gagal membaca Mic!");
        }
      }
    }
    
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

void extractMelSpec() {
  float sc = inputTensor->params.scale;
  int zp = inputTensor->params.zero_point;
  int8_t *d = inputTensor->data.int8;

  for (int frame = 0; frame < N_FRAMES; frame++) {
    int start = frame * HOP_LEN;
    
    // 1. Hitung rata-rata (mean) untuk DC offset removal
    float sum = 0.0f;
    for (int k = 0; k < FFT_N; k++) {
      int curr = (historyHead + start + k) % AUDIO_LEN;
      int prev = (curr == 0) ? (AUDIO_LEN - 1) : (curr - 1);
      int next = (curr + 1) % AUDIO_LEN;
      
      // Low Pass Filter sederhana
      float sample = (historyBuffer[prev] + historyBuffer[curr] + historyBuffer[next]) / (3.0f * 32768.0f);
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
      float std_m = pgm_read_float(&MEL_STD[m]);
      float feat = (log_mel - mean_m) / std_m;
      
      // Quantize and write DIRECTLY to TFLite Input Tensor (BYPASS inputFlat)
      int q = (int)roundf(feat / sc) + zp;
      d[frame * N_MELS + m] = (int8_t)constrain(q, -128, 127);
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
  // Fitur sudah dikuantisasi dan berada di inputTensor->data.int8 lewat extractMelSpec()

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

// ── Variabel EMA State ──
static float ema_probs[NUM_CLASSES] = {0.0f, 0.0f, 1.0f, 0.0f};
#define EMA_ALPHA 0.15f
#define SIREN_THRESHOLD 0.80f
#define OVERRIDE_THRESHOLD 0.95f
static int locked_class_idx = 2; // Default NORMAL (2)

int smartDetectEMA(float &out_prob, bool &is_thinking) {
  // 1. Update EMA
  for(int i=0; i<NUM_CLASSES; i++) {
    ema_probs[i] = (1.0f - EMA_ALPHA) * ema_probs[i] + EMA_ALPHA * outputScores[i];
  }
  
  // 2. Cari Best Probability
  int best_idx = 0;
  float best_prob = ema_probs[0];
  for(int i=1; i<NUM_CLASSES; i++) {
    if(ema_probs[i] > best_prob) {
      best_prob = ema_probs[i];
      best_idx = i;
    }
  }
  
  is_thinking = false;
  
  // 3. Logika Lock-on
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
    if (best_idx != 2 && best_idx != locked_class_idx && best_prob >= OVERRIDE_THRESHOLD) {
      locked_class_idx = best_idx;
      for(int i=0; i<NUM_CLASSES; i++) ema_probs[i] = 0.0f;
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

// Menggambar frame HUD modern futuristik untuk layar kotak (X:0-239, Y:0-319)
void drawHUDFrame(uint16_t color) {
  // Kotak garis luar tipis abu-abu gelap
  tft.drawRect(8, 8, 224, 304, 0x2104);
  
  // Siku sudut Top-Left (Tebal 3px, panjang 15px)
  tft.fillRect(6, 6, 15, 3, color);
  tft.fillRect(6, 6, 3, 15, color);
  
  // Siku sudut Top-Right
  tft.fillRect(219, 6, 15, 3, color);
  tft.fillRect(231, 6, 3, 15, color);
  
  // Siku sudut Bottom-Left
  tft.fillRect(6, 311, 15, 3, color);
  tft.fillRect(6, 299, 3, 15, color);
  
  // Siku sudut Bottom-Right
  tft.fillRect(219, 311, 15, 3, color);
  tft.fillRect(231, 299, 3, 15, color);
}

// Menggambar ikon Power Button grafis ⏻ di (120, 155)
void drawPowerIcon(uint16_t color) {
  // Lingkaran utama terputus (radius 25)
  tft.drawCircle(120, 155, 25, color);
  tft.drawCircle(120, 155, 24, color); // Tebalkan sedikit
  // Hapus celah bagian atas (sekitar -45 sampai 45 derajat)
  tft.fillRect(110, 125, 20, 12, ST77XX_BLACK);
  // Garis vertikal tebal di tengah celah
  tft.fillRect(119, 118, 3, 22, color);
}

// Menggambar ikon grafis megah untuk alert sirine
void drawWarningIcon(int cls, uint16_t color) {
  // Titik tengah X = 120, Y = 130
  switch (cls) {
  case 0: // AMBULANCE: Medical Cross
    tft.fillRect(105, 125, 30, 10, color);
    tft.fillRect(115, 115, 10, 30, color);
    break;
  case 1: // FIRETRUCK: Warning Triangle + Exclamation
    tft.drawTriangle(120, 110, 95, 145, 145, 145, color);
    tft.fillRect(119, 120, 3, 15, color);
    tft.fillRect(119, 139, 3, 3, color);
    break;
  case 3: // POLICE: Shield/Badge
    tft.drawLine(105, 115, 135, 115, color);
    tft.drawLine(135, 115, 135, 135, color);
    tft.drawLine(135, 135, 120, 150, color);
    tft.drawLine(120, 150, 105, 135, color);
    tft.drawLine(105, 135, 105, 115, color);
    tft.fillCircle(120, 130, 4, color);
    break;
  }
}

void lcdShowDetection(int cls, float conf, bool thinking) {
  if (lcdMutex == NULL || xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(100)) != pdTRUE) return;
  
  // Modifikasi agar state thinking dianggap state yang berbeda secara UI
  int stateID = thinking ? (cls + 10) : cls; 
  
  if (stateID != lastShownClass) {
    lastShownClass = stateID;
    lastShownConf = conf;
    strobeToggle = true; // Mulai dengan warna menyala

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
    drawHUDFrame(fg);

    if (cls == 2 && !thinking) {
      // Tampilan LISTENING / SCANNING
      tft.setTextSize(2);
      tft.setTextColor(ST77XX_GREEN);
      tft.setCursor((240 - (12 * 12)) / 2, 80); // "SIREN MASTER"
      tft.print("SIREN MASTER");

      tft.setCursor((240 - (11 * 12)) / 2, 225); // "SCANNING..."
      tft.print("SCANNING...");
    } else if (thinking) {
      // Tampilan MENILAI...
      tft.setTextSize(2);
      tft.setTextColor(fg);
      tft.setCursor((240 - (13 * 12)) / 2, 65); // "? MENILAI ?"
      tft.print("? MENILAI ?");

      // Ikon berpikir sementara (kotak kecil)
      tft.fillRect(110, 120, 20, 20, fg);

      tft.setTextSize(2);
      int textW = strlen(alertText) * 12;
      tft.setCursor((240 - textW) / 2, 180);
      tft.print(alertText);

      tft.setTextSize(2);
      tft.setCursor((240 - (10 * 12)) / 2, 225);
      tft.printf("Conf: %3.0f%%", conf * 100);
    } else {
      // Tampilan ALERT Megah
      tft.setTextSize(2);
      tft.setTextColor(fg);
      tft.setCursor((240 - (13 * 12)) / 2, 65); // "!! WARNING !!"
      tft.print("!! WARNING !!");

      // Gambar Ikon Grafis Kustom
      drawWarningIcon(cls, fg);

      tft.setTextSize(3);
      int textW = strlen(alertText) * 18;
      tft.setCursor((240 - textW) / 2, 180);
      tft.print(alertText);

      tft.setTextSize(2);
      tft.setCursor((240 - (10 * 12)) / 2, 225); // Centered "Conf: XX%"
      tft.printf("Conf: %3.0f%%", conf * 100);
    }
  } else if (cls != 2 || thinking) {
    lastShownConf = conf;
  }
  xSemaphoreGive(lcdMutex);
}

void applyOutputs(int cls, float conf, bool thinking) {
  // Cegah AI menyalakan komponen jika sistem baru saja dimatikan
  if (!btnManager.isActive()) return;
  
  if (thinking || cls == 2) {
    // Sedang MENILAI atau SAFE: Jangan bunyikan sirine/motor dulu!
    setRGB(0, 0, 0);
    motorPattern = 0;
  } else {
    // ALERT PENUH: Bunyikan dan nyalakan lampu!
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
    
    // Tampilan Standby Mewah Komersial
    tft.fillScreen(ST77XX_BLACK);
    drawHUDFrame(0x4208); // Frame loading/standby abu-abu redup
    
    tft.setTextColor(0x4208); // Abu-abu redup
    tft.setTextSize(2);
    tft.setCursor((240 - (12 * 12)) / 2, 80);
    tft.print("SIREN MASTER");
    
    // Gambar Power Icon merah redup
    drawPowerIcon(0x8000); 
    
    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(2);
    tft.setCursor((240 - (7 * 12)) / 2, 205);
    tft.print("STANDBY");
    
    tft.setTextColor(0x4208); // Abu-abu redup
    tft.setTextSize(1);
    tft.setCursor((240 - (19 * 6)) / 2, 230);
    tft.print("[ Press to Wake ]");
    
    delay(1000); // Tampilkan standby screen selama 1 detik
    
    // 3. Matikan backlight layar
    digitalWrite(TFT_BL, LOW);
    
    if (lcdMutex != NULL) {
      xSemaphoreGive(lcdMutex);
    }
    
    // 4. MATIKAN ESP32 TOTAL via Deep Sleep
    //    ESP32 akan MATI TOTAL. Saat tombol ditekan lagi,
    //    ESP32 akan REBOOT dari awal (setup()) seperti baru dinyalakan.
    btnManager.shutdown(PIN_BUTTON);
    // --- Kode di bawah ini TIDAK AKAN PERNAH dieksekusi ---
  }
}

// Menggambar gelombang live oscilloscope ketika mode SAFE
void updateLiveVisualizer() {
  if (!btnManager.isActive()) return;
  if (lastShownClass != 2) return; // Hanya gambar ketika mode SAFE
  
  unsigned long now = millis();
  if (now - lastVisualizerUpdate < 50) return; // 20 FPS
  
  if (lcdMutex == NULL || xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(20)) != pdTRUE) return;
  lastVisualizerUpdate = now;
  
  // Haluskan pergerakan amplitudo ke arah target
  currentAmplitude = currentAmplitude * 0.8f + targetAmplitude * 0.2f;
  
  // Hapus area gelombang lama (Y=120 sampai Y=200, pusat Y=160)
  tft.fillRect(35, 120, 170, 80, ST77XX_BLACK);
  
  // Gambarkan garis tengah titik-titik (Zero Centerline Dotted Grid)
  for (int x = 40; x <= 200; x += 6) {
    tft.drawPixel(x, 160, 0x01E0); // Faint green dotted line
  }
  
  int prevX = 35;
  int prevY1 = 160;
  int prevY2 = 160;
  visualizerPhase += 0.25f; // Kecepatan gerak gelombang
  
  for (int x = 35; x <= 205; x += 3) {
    float angle = (x - 35) * 0.08f - visualizerPhase;
    int offset = (int)(currentAmplitude * sin(angle));
    int y1 = 160 + offset;
    int y2 = 160 - offset; // Gelombang cermin
    if (x > 35) {
      tft.drawLine(prevX, prevY1, x, y1, 0x07E0); // Hijau Neon
      tft.drawLine(prevX, prevY2, x, y2, 0x0700); // Hijau Gelap (hologram)
    }
    prevX = x;
    prevY1 = y1;
    prevY2 = y2;
  }
  xSemaphoreGive(lcdMutex);
}

// Berkedip strobo ketika mendeteksi sirine
void updateAlertStrobe() {
  if (!btnManager.isActive()) return;
  if (lastShownClass == 2 || lastShownClass == -1) return; // Jangan strobo jika SAFE
  
  unsigned long now = millis();
  if (now - lastStrobeUpdate < 200) return; // Strobo 5Hz (200ms sekali)
  
  if (lcdMutex == NULL || xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(20)) != pdTRUE) return;
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
    setRGB(strobeToggle ? 1 : 0, strobeToggle ? 1 : 0, 0); // Strobo LED Kuning Sinkron
    break;
  case 3:
    bg = strobeToggle ? ST77XX_BLUE : ST77XX_BLACK;
    fg = ST77XX_WHITE;
    alertText = "POLICE";
    setRGB(0, 0, strobeToggle ? 1 : 0); // Strobo LED Biru Sinkron
    break;
  }
  
  tft.fillScreen(bg);
  drawHUDFrame(fg);
  
  tft.setTextSize(2);
  tft.setTextColor(fg);
  tft.setCursor((240 - (13 * 12)) / 2, 65);
  tft.print("!! WARNING !!");
  
  // Gambar Ikon Strobo
  drawWarningIcon(lastShownClass, fg);
  
  tft.setTextSize(3);
  int textW = strlen(alertText) * 18;
  tft.setCursor((240 - textW) / 2, 180);
  tft.print(alertText);
  
  tft.setTextSize(2);
  tft.setCursor((240 - (10 * 12)) / 2, 225);
  tft.printf("Conf: %3.0f%%", lastShownConf * 100);
  
  xSemaphoreGive(lcdMutex);
}

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  digitalWrite(LED_R_PIN, r);
  digitalWrite(LED_G_PIN, g);
  digitalWrite(LED_B_PIN, b);
}

void updateMotor() {
  if (motorPattern == 0) {
    digitalWrite(MOTOR_PIN, LOW);
    return;
  }
  unsigned long now = millis();
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
  for(;;) {
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
  i2s_config_t i2s_config = {.mode =
                                 (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
                             .sample_rate = SAMPLE_RATE_HZ,
                             .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
                             .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
#if ESP_ARDUINO_VERSION_MAJOR >= 3
                             .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#else
                             .communication_format =
                                  (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S |
                                                      I2S_COMM_FORMAT_I2S_MSB),
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
    Serial.printf("[ERROR] I2S Install Gagal! Err1: %d, Err2: %d\n", err1, err2);
  } else {
    Serial.println("[OK] I2S Mic Berhasil Diinstall!");
  }

  // LCD
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
  delay(200); // Beri waktu LCD untuk stabil setelah dinyalakan
  tft.setSPISpeed(8000000); // 8MHz (pelan tapi stabil untuk breadboard)
  tft.init(240, 320, SPI_MODE3); // Inisialisasi layar 240x320 (Full Screen!)
  tft.setRotation(2); // Putar layar agar tulisan tidak terbalik/miring
  
  // SPLASH SCREEN STARTUP MEWAH
  tft.fillScreen(ST77XX_BLACK);
  drawHUDFrame(ST77XX_CYAN);
  
  tft.setTextColor(ST77XX_CYAN);
  tft.setTextSize(2);
  tft.setCursor((240 - (12 * 12)) / 2, 80); // "SIREN MASTER"
  tft.println("SIREN MASTER");
  
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor((240 - (15 * 6)) / 2, 125);
  tft.println("SYSTEM STARTING");
  
  // Progress Bar Container
  tft.drawRect(55, 150, 130, 8, ST77XX_CYAN);
  tft.fillRect(57, 152, 15, 4, ST77XX_CYAN); // 10%
  tft.setCursor((240 - (14 * 6)) / 2, 175);
  tft.println("Allocating RAM");

  // ── ALOKASI MEMORI MURNI RAM INTERNAL (WROOM SAFE) ──
  tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  chunkBuffer = (int16_t*)malloc(SAMPLE_RATE * sizeof(int16_t));
  historyBuffer = (int16_t*)malloc(AUDIO_LEN * sizeof(int16_t));

  if (!tensor_arena || !chunkBuffer || !historyBuffer) {
    Serial.println("[ERROR] GAGAL Mengalokasikan Memori! Alat berhenti.");
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(30, 175);
    tft.println("OUT OF MEMORY!");
    while(1) delay(100);
  }
  memset(historyBuffer, 0, AUDIO_LEN * sizeof(int16_t));

  // 40% Progress
  tft.fillRect(57, 152, 50, 4, ST77XX_CYAN);
  tft.fillRect(20, 175, 200, 10, ST77XX_BLACK); // clear text
  tft.setCursor((240 - (16 * 6)) / 2, 175);
  tft.println("Loading AI Model");

  // ── TFLite: Load model ──
  tflModel = tflite::GetModel(siren_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model version mismatch!");
    while (1)
      delay(1000);
  }

  // 60% Progress
  tft.fillRect(57, 152, 80, 4, ST77XX_CYAN);
  tft.fillRect(20, 175, 200, 10, ST77XX_BLACK);
  tft.setCursor((240 - (19 * 6)) / 2, 175);
  tft.println("Registering Kernels");

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
  tft.fillRect(20, 175, 200, 10, ST77XX_BLACK);
  tft.setCursor((240 - (17 * 6)) / 2, 175);
  tft.println("Allocating Tensors");

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
  Serial.printf("[IN]  type=%d shape=(%d,%d,%d,%d)\n", inputTensor->type,
                inputTensor->dims->data[0], inputTensor->dims->data[1],
                inputTensor->dims->data[2], inputTensor->dims->data[3]);
  Serial.printf("[OUT] type=%d shape=(%d,%d)\n", outputTensor->type,
                outputTensor->dims->data[0], outputTensor->dims->data[1]);

  // 100% Progress
  tft.fillRect(57, 152, 126, 4, ST77XX_CYAN);
  tft.fillRect(20, 175, 200, 10, ST77XX_BLACK);
  tft.setCursor((240 - (12 * 6)) / 2, 175);
  tft.println("System Ready");
  delay(300);

  // Startup test (LED & Haptic Pulse)
  setRGB(0, 1, 0); // Hijau
  digitalWrite(MOTOR_PIN, HIGH);
  
  tft.fillScreen(ST77XX_BLACK);
  
  // 1. Gambar frame HUD untuk panel kotak
  drawHUDFrame(ST77XX_GREEN);
  
  // 2. Animasi Laser Scan Line Futuristik untuk startup panel kotak
  for (int y = 10; y < 310; y += 5) {
    tft.drawFastHLine(10, y, 220, ST77XX_CYAN);
    delay(8);
    tft.drawFastHLine(10, y, 220, ST77XX_BLACK);
    drawHUDFrame(0x4208); // Frame loading abu-abu redup
  }
  drawHUDFrame(ST77XX_GREEN);
  
  tft.setTextColor(ST77XX_GREEN);
  tft.setTextSize(2);
  tft.setCursor((240 - (12 * 12)) / 2, 80); // "SIREN MASTER"
  tft.println("SIREN MASTER");
  
  tft.setTextSize(3);
  tft.setCursor((240 - (5 * 18)) / 2, 140); // "READY"
  tft.println("READY");
  
  tft.setTextSize(2);
  tft.setCursor((240 - (11 * 12)) / 2, 225); // "SCANNING..."
  tft.println("SCANNING...");
  
  delay(1000);
  digitalWrite(MOTOR_PIN, LOW);
  setRGB(0, 0, 0);
  delay(500);

  motorTimer = millis();
  
  // Start FreeRTOS Tasks
  audioSemaphore = xSemaphoreCreateBinary();
  lcdMutex = xSemaphoreCreateMutex();
  
  BaseType_t res1 = xTaskCreatePinnedToCore(
    audioTaskCode, "TaskAudio", 4096, NULL, 2, &TaskAudio, 0); // Core 0 untuk I2S
    
  BaseType_t res2 = xTaskCreatePinnedToCore(
    inferenceTaskCode, "TaskInference", 8192, NULL, 1, &TaskInference, 1); // Core 1 untuk AI
    
  BaseType_t res3 = xTaskCreatePinnedToCore(
    ioTaskCode, "TaskIO", 4096, NULL, 2, NULL, 1); // Core 1, Priority 2 (Preempts AI) - Stack diperbesar ke 4096

  if (res1 != pdPASS || res2 != pdPASS || res3 != pdPASS) {
    Serial.println("[FATAL ERROR] Gagal membuat FreeRTOS Task! Kehabisan Memori RAM.");
    tft.fillScreen(ST77XX_RED);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(10, 175);
    tft.println("TASK ERROR!");
    while(1) delay(100);
  }
}

// ═══════════════════════════════════════════════════════════════
// LOOP
// ═══════════════════════════════════════════════════════════════
void inferenceTaskCode(void *pvParameters) {
  for(;;) {
    if (xSemaphoreTake(audioSemaphore, portMAX_DELAY) == pdTRUE) {
      if (!btnManager.isActive()) continue;
      
      // --- NOISE GATE ---
      float sumSq = 0;
      for(int i = 0; i < AUDIO_LEN; i++) {
        sumSq += ((float)historyBuffer[i] * historyBuffer[i]);
      }
      float rms = sqrt(sumSq / AUDIO_LEN);
      
      // Menurunkan batas hening ke 1400.0f agar AI lebih sensitif mendeteksi sirine luar dari dalam mobil/ruangan (noise lantai listrik mic di 1100-1260)
      if (rms < 1400.0f) {
        Serial.printf("[ZzZ] Hening (RMS: %.1f) -> SAFE\n\n", rms);
        targetAmplitude = 0.0f; // Gelombang lurus
        
        // Force NORMAL scores during silence to allow EMA to decay properly
        for(int i=0; i<NUM_CLASSES; i++) outputScores[i] = 0.0f;
        outputScores[2] = 1.0f;
        
        bool thinking = false;
        float final_prob = 1.0f;
        int cls = smartDetectEMA(final_prob, thinking);
        applyOutputs(cls, final_prob, thinking);
        continue;
      }

      // Atur ketinggian gelombang visualizer berdasarkan volume suara
      targetAmplitude = constrain((rms - 1400.0f) * 0.02f, 5.0f, 35.0f);

      Serial.println("[DSP] Mel-Spectrogram...");
      extractMelSpec();

      Serial.println("[ML] Inference...");
      float conf = 0;
      runInference(conf); // Updates outputScores[] internally
      
      bool thinking = false;
      float final_prob = 0.0f;
      int cls = smartDetectEMA(final_prob, thinking);
      applyOutputs(cls, final_prob, thinking);

      // Cetak hasil klasifikasi berserta volume RMS untuk mempermudah monitoring/kalibrasi
      Serial.printf("[Result] %s | %.1f%% %s (RMS: %.1f)\n\n", CLASS_LABELS[cls], final_prob * 100, thinking ? "(MENILAI...)" : "", rms);
    }
  }
}

void loop() {
  // Semua tugas sudah ditangani oleh FreeRTOS Tasks
  vTaskDelay(pdMS_TO_TICKS(1000));
}
