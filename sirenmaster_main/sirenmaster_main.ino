/*
 * sirenmaster_main.ino
 * TinyML Emergency Vehicle Siren Detector — ESP32/ESP32-S3
 * Menggunakan TensorFlow Lite Micro (Chirale_TensorFlowLite)
 *
 * WIRING:
 * INMP441: WS→32, SCK→33, SD→35
 * Motor Getar → GPIO25
 * LED R/G/B  → GPIO26 / GPIO27 / GPIO21  (PWM-capable)
 * ST7789: SCK→18, MOSI→23, CS→14, DC→22, RST→4, BL→15
 * Push Button → GPIO12
 *
 * Libraries: Chirale_TensorFlowLite, Adafruit_ST7789, Adafruit_GFX
 */

#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <Arduino.h>
#include <SPI.h>
#include <driver/i2s.h>
// PWM LED RGB via Arduino Core v3.x pin-based LEDC API

// ── TensorFlow Lite Micro (Chirale v2.0) ─────────────────────
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Chirale_TensorFlowLite.h>

#include "buttonmanager.h"
#include "model.h"


// ── Pin Definitions ──────────────────────────────────────────
// INMP441 Mic (I2S)
#define I2S_WS_PIN  32
#define I2S_SCK_PIN 33
#define I2S_SD_PIN  35

// Vibration Motor
#define MOTOR_PIN   25

// LED RGB (PWM via LEDC)
#define LED_R_PIN   26
#define LED_G_PIN   27
#define LED_B_PIN   21

// LEDC PWM config (pin-based API, Core v3.x)
#define LEDC_FREQ   5000   // 5kHz PWM
#define LEDC_RES    8      // 8-bit (0-255)

// LCD ST7789 (SPI)
#define LCD_SCL_PIN 18
#define LCD_SDA_PIN 23
#define LCD_RES_PIN 4
#define LCD_DC_PIN  22
#define LCD_CS_PIN  14

// Others
#define PIN_BUTTON  12
#define TFT_BL      15

// ── Audio ─────────────────────────────────────────────────────
#define SAMPLE_RATE SAMPLE_RATE_HZ
#define AUDIO_LEN   AUDIO_SAMPLES
#define FFT_N       N_FFT_SIZE
#define FFT_BINS    N_FFT_BINS
#define HOP_LEN     N_HOP_LENGTH
#define N_FRAMES    N_TIME_FRAMES
#define N_MELS      N_MEL_FILTERS

// ── Microphone Calibration ───────────────────────────────────
#define MIC_GAIN    2.5f

// ── Inference ────────────────────────────────────────────────
#define CONFIDENCE_THR    0.92f
#define TENSOR_ARENA_KB   85

const char *CLASS_LABELS[NUM_CLASSES] = {"AMBULANCE", "FIRETRUCK", "NOISE", "POLICE"};

// ═══════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════
Adafruit_ST7789 tft = Adafruit_ST7789(LCD_CS_PIN, LCD_DC_PIN, LCD_RES_PIN);

// TFLite Micro
static tflite::MicroMutableOpResolver<14> resolver;
static const tflite::Model *tflModel      = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *inputTensor  = nullptr;
static TfLiteTensor *outputTensor = nullptr;

constexpr int kTensorArenaSize = TENSOR_ARENA_KB * 1024;
static uint8_t *tensor_arena = nullptr;

// Audio & DSP buffers
static int16_t *chunkBuffer   = nullptr;
static int16_t *historyBuffer = nullptr;
static int historyHead = 0;
SemaphoreHandle_t audioSemaphore;
TaskHandle_t TaskAudio;
TaskHandle_t TaskInference;
static float fftReal[FFT_N];
static float fftImag[FFT_N];
static float melEnergy[N_MELS];
static float outputScores[NUM_CLASSES];
static float powerSpec[FFT_BINS];

// Motor
static unsigned long motorTimer   = 0;
static int motorState   = 0;
static int motorPattern = 0;

// Button
ButtonManager btnManager(PIN_BUTTON);
static SemaphoreHandle_t lcdMutex = NULL;

// Smart detect — EMA
static float ema_probs[NUM_CLASSES] = {0.0f, 0.0f, 1.0f, 0.0f};
#define EMA_ALPHA         0.15f
#define SIREN_THRESHOLD   0.80f
#define OVERRIDE_THRESHOLD 0.95f
static int locked_class_idx = 2;

// Adaptive Noise Floor
static float noise_floor = 400.0f;
#define NOISE_FLOOR_MIN   200.0f
#define NOISE_FLOOR_MAX   1200.0f
#define NOISE_FLOOR_ALPHA 0.005f

// LCD strobe state (non-blocking, tidak pakai fillScreen berulang)
static int  lastShownClass = -1;
static float lastShownConf = -1.0f;
static bool strobeToggle   = false;
static unsigned long lastStrobeUpdate = 0;

// Shared RGB state (agar strobe & applyOutputs tidak konflik)
static volatile uint8_t rgbR = 0, rgbG = 0, rgbB = 0;
static SemaphoreHandle_t rgbMutex = NULL;

// Visualizer
static float targetAmplitude  = 0;


// ═══════════════════════════════════════════════════════════════
// RGB PWM — Smooth, tidak flicker
// ═══════════════════════════════════════════════════════════════

void setupRGB() {
  // ESP32 Core v3.x: pin-based LEDC API (ledcSetup/ledcAttachPin dihapus)
  ledcAttach(LED_R_PIN, LEDC_FREQ, LEDC_RES);
  ledcAttach(LED_G_PIN, LEDC_FREQ, LEDC_RES);
  ledcAttach(LED_B_PIN, LEDC_FREQ, LEDC_RES);
  ledcWrite(LED_R_PIN, 0);
  ledcWrite(LED_G_PIN, 0);
  ledcWrite(LED_B_PIN, 0);
}

// Tulis RGB dalam nilai 0-255 (pin-based)
void setRGBPWM(uint8_t r, uint8_t g, uint8_t b) {
  ledcWrite(LED_R_PIN, r);
  ledcWrite(LED_G_PIN, g);
  ledcWrite(LED_B_PIN, b);
}

// Wrapper boolean untuk kompatibilitas kode lama (1 = ON, 0 = OFF)
void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  setRGBPWM(r ? 200 : 0, g ? 200 : 0, b ? 200 : 0);
}

// Set RGB dengan mutex aman dari dua task berbeda
void setRGBSafe(uint8_t r, uint8_t g, uint8_t b) {
  if (rgbMutex && xSemaphoreTake(rgbMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    rgbR = r; rgbG = g; rgbB = b;
    setRGBPWM(r, g, b);
    xSemaphoreGive(rgbMutex);
  }
}


// ═══════════════════════════════════════════════════════════════
// MOTOR — Non-blocking, pattern per kelas
// ═══════════════════════════════════════════════════════════════

void updateMotor() {
  static int lastMotorPattern = -1;

  if (motorPattern == 0) {
    if (lastMotorPattern != 0) {
      digitalWrite(MOTOR_PIN, LOW);
      motorState      = 0;
      lastMotorPattern = 0;
    }
    return;
  }

  unsigned long now = millis();

  // Pola baru → nyalakan motor INSTAN tanpa tunggu timer
  if (motorPattern != lastMotorPattern) {
    lastMotorPattern = motorPattern;
    motorState  = 1;
    motorTimer  = now;
    digitalWrite(MOTOR_PIN, HIGH);
    return;
  }

  // Pattern 1: AMBULANCE — denyut cepat 100ms ON/OFF
  if (motorPattern == 1) {
    if (now - motorTimer >= 100) {
      motorState ^= 1;
      motorTimer  = now;
      digitalWrite(MOTOR_PIN, motorState);
    }
  }
  // Pattern 2: FIRETRUCK — 300ms ON / 200ms OFF (lebih tegas)
  else if (motorPattern == 2) {
    uint32_t interval = motorState ? 300 : 200;
    if (now - motorTimer >= interval) {
      motorState ^= 1;
      motorTimer  = now;
      digitalWrite(MOTOR_PIN, motorState);
    }
  }
  // Pattern 3: POLICE — 3x ketuk cepat lalu jeda (tit-tit-tit ... pause)
  else if (motorPattern == 3) {
    static int step = 0;
    // ON 120ms, OFF 120ms, ON 120ms, OFF 120ms, ON 120ms, OFF 600ms pause
    const uint32_t iv[]    = {120, 120, 120, 120, 120, 600};
    const uint8_t  state[] = { 1,   0,   1,   0,   1,   0 };
    const int SEQ_LEN = 6;
    if (now - motorTimer >= iv[step]) {
      step = (step + 1) % SEQ_LEN;
      motorState = state[step];
      motorTimer = now;
      digitalWrite(MOTOR_PIN, motorState);
    }
  }
}


// ═══════════════════════════════════════════════════════════════
// AUDIO
// ═══════════════════════════════════════════════════════════════
void audioTaskCode(void *pvParameters) {
  const int chunk_size     = SAMPLE_RATE;
  const int raw_chunk_size = 512;
  size_t bytesIn = 0;
  int32_t *raw_buf = (int32_t *)malloc(raw_chunk_size * sizeof(int32_t));
  int chunk_idx = 0;

  for (;;) {
    if (!btnManager.isActive()) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    while (chunk_idx < chunk_size) {
      esp_err_t result = i2s_read(I2S_NUM_0, raw_buf,
                                  raw_chunk_size * 4, &bytesIn,
                                  pdMS_TO_TICKS(100));
      int samples_read = bytesIn / 4;

      if (result == ESP_OK && samples_read > 0) {
        for (int j = 0; j < samples_read - 1; j += 2) {
          if (chunk_idx >= chunk_size) break;
          int16_t s1 = (int16_t)(raw_buf[j]     >> 16);
          int16_t s2 = (int16_t)(raw_buf[j + 1] >> 16);
          float boosted = (((float)s1 + s2) / 2.0f) * MIC_GAIN;
          chunkBuffer[chunk_idx++] = (int16_t)constrain(boosted, -32768.0f, 32767.0f);
        }
      } else {
        while (chunk_idx < chunk_size) chunkBuffer[chunk_idx++] = 0;
        Serial.println("[ERROR] i2s_read Timeout / Mic Error!");
      }
    }
    chunk_idx = 0;

    for (int i = 0; i < chunk_size; i++) {
      historyBuffer[historyHead] = chunkBuffer[i];
      historyHead = (historyHead + 1) % AUDIO_LEN;
    }
    xSemaphoreGive(audioSemaphore);
  }
}


// ═══════════════════════════════════════════════════════════════
// FFT & MEL
// ═══════════════════════════════════════════════════════════════
void computeFFT(float *vR, float *vI, int n) {
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float tr = vR[i]; vR[i] = vR[j]; vR[j] = tr;
      float ti = vI[i]; vI[i] = vI[j]; vI[j] = ti;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * PI / (float)len;
    float wRe = cosf(ang), wIm = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cuRe = 1.0f, cuIm = 0.0f;
      for (int k = 0; k < (len >> 1); k++) {
        float uRe = vR[i + k], uIm = vI[i + k];
        float vRe = vR[i + k + (len >> 1)] * cuRe - vI[i + k + (len >> 1)] * cuIm;
        float vIm = vR[i + k + (len >> 1)] * cuIm + vI[i + k + (len >> 1)] * cuRe;
        vR[i + k] = uRe + vRe; vI[i + k] = uIm + vIm;
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
  bool isFloat32Input = (inputTensor->type == kTfLiteFloat32);
  float *df = isFloat32Input ? inputTensor->data.f    : nullptr;
  int8_t *di = isFloat32Input ? nullptr : inputTensor->data.int8;
  float sc   = isFloat32Input ? 1.0f   : inputTensor->params.scale;
  int   zp   = isFloat32Input ? 0      : inputTensor->params.zero_point;

  int active_samples = active_chunks * SAMPLE_RATE;
  if (active_samples > AUDIO_LEN) active_samples = AUDIO_LEN;
  if (active_samples < SAMPLE_RATE) active_samples = SAMPLE_RATE;

  float max_val = 0.0f;
  for (int idx = 0; idx < active_samples; idx++) {
    int tci  = (AUDIO_LEN - active_samples) + (idx % active_samples);
    int curr = (historyHead + tci) % AUDIO_LEN;
    float val = fabsf((float)historyBuffer[curr]);
    if (val > max_val) max_val = val;
  }
  float gain = 1.0f;
  if (max_val > 1.0f) {
    gain = 32767.0f / max_val;
    if (gain > 10.0f) gain = 10.0f;
  }

  for (int frame = 0; frame < N_FRAMES; frame++) {
    int start = frame * HOP_LEN;
    float sum = 0.0f;
    for (int k = 0; k < FFT_N; k++) {
      int idx  = start + k;
      int tci  = (AUDIO_LEN - active_samples) + (idx % active_samples);
      int curr = (historyHead + tci) % AUDIO_LEN;
      int prev = (curr == 0) ? (AUDIO_LEN - 1) : (curr - 1);
      int next = (curr + 1) % AUDIO_LEN;
      float sample = ((float)historyBuffer[prev] + historyBuffer[curr] + historyBuffer[next])
                     / (3.0f * 32768.0f) * gain;
      fftReal[k] = sample;
      sum += sample;
    }
    float mean = sum / FFT_N;
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
      float mean_m  = pgm_read_float(&MEL_MEAN[m]);
      float std_m   = pgm_read_float(&MEL_STD[m]);
      float feat    = (log_mel - mean_m) / std_m;
      int tensor_idx = frame * N_MELS + m;
      if (isFloat32Input) {
        df[tensor_idx] = feat;
      } else {
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
  if (!interpreter) { confidence = 0; return 2; }
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("[ERROR] Invoke gagal!");
    confidence = 0; return 2;
  }
  if (outputTensor->type == kTfLiteInt8) {
    float sc = outputTensor->params.scale;
    int   zp = outputTensor->params.zero_point;
    for (int i = 0; i < NUM_CLASSES; i++)
      outputScores[i] = ((float)outputTensor->data.int8[i] - zp) * sc;
  } else {
    for (int i = 0; i < NUM_CLASSES; i++)
      outputScores[i] = outputTensor->data.f[i];
  }
  int best = 0; float bestS = outputScores[0];
  for (int i = 1; i < NUM_CLASSES; i++)
    if (outputScores[i] > bestS) { bestS = outputScores[i]; best = i; }
  confidence = bestS;
  return (bestS >= CONFIDENCE_THR) ? best : 2;
}

// ── EMA Smart Detect ──────────────────────────────────────────
int smartDetectEMA(float &out_prob, bool &is_thinking) {
  float clean_scores[NUM_CLASSES], score_sum = 0.0f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    clean_scores[i] = outputScores[i] < 0.0f ? 0.0f : outputScores[i];
    score_sum += clean_scores[i];
  }
  if (score_sum > 0.0f)
    for (int i = 0; i < NUM_CLASSES; i++) clean_scores[i] /= score_sum;
  else {
    for (int i = 0; i < NUM_CLASSES; i++) clean_scores[i] = (i == 2) ? 1.0f : 0.0f;
  }

  for (int i = 0; i < NUM_CLASSES; i++) {
    float alpha = EMA_ALPHA;
    // Percepat 3x jika AI sangat yakin (>85%) dan bukan NORMAL
    if (clean_scores[i] > 0.85f && i != 2) alpha = 0.50f;
    ema_probs[i] = (1.0f - alpha) * ema_probs[i] + alpha * clean_scores[i];
  }

  float ema_sum = 0.0f;
  for (int i = 0; i < NUM_CLASSES; i++) ema_sum += ema_probs[i];
  if (ema_sum > 0.0f)
    for (int i = 0; i < NUM_CLASSES; i++) ema_probs[i] /= ema_sum;

  int best_idx = 0; float best_prob = ema_probs[0];
  for (int i = 1; i < NUM_CLASSES; i++)
    if (ema_probs[i] > best_prob) { best_prob = ema_probs[i]; best_idx = i; }

  is_thinking = false;

  if (locked_class_idx != 2 && ema_probs[locked_class_idx] < SIREN_THRESHOLD)
    locked_class_idx = 2;

  if (locked_class_idx == 2) {
    if (best_idx != 2 && best_prob >= SIREN_THRESHOLD) {
      locked_class_idx = best_idx;
      out_prob = best_prob;
    } else if (best_idx != 2 && best_prob > 0.40f) {
      is_thinking = true;
      out_prob = best_prob;
      return best_idx;
    } else {
      out_prob = ema_probs[2];
      return 2;
    }
  } else {
    if (best_idx != 2 && best_idx != locked_class_idx && best_prob >= OVERRIDE_THRESHOLD) {
      locked_class_idx = best_idx;
      for (int i = 0; i < NUM_CLASSES; i++) ema_probs[i] = 0.0f;
      ema_probs[best_idx] = 1.0f;
      out_prob = best_prob;
    } else {
      out_prob = ema_probs[locked_class_idx];
    }
  }
  return locked_class_idx;
}


// ═══════════════════════════════════════════════════════════════
// LCD — Render sekali per state change, update conf on-the-fly
// ═══════════════════════════════════════════════════════════════
void drawCenteredText(const char *text, int y, int sz, uint16_t color) {
  tft.setTextSize(sz);
  tft.setTextColor(color);
  int len = strlen(text);
  int x   = (240 - (len * 6 * sz)) / 2;
  if (x < 0) x = 0;
  tft.setCursor(x, y);
  tft.print(text);
}

// Gambar layar lengkap untuk state tertentu (dipanggil sekali saat state berubah)
void lcdDrawFullState(int cls, float conf, bool thinking) {
  uint16_t bg = ST77XX_BLACK;
  uint16_t fg = ST77XX_WHITE;
  const char *alertText = "";

  if (thinking) {
    bg = 0x8200;  // Merah tua
    fg = 0xFD20;  // Oranye kuning
    alertText = CLASS_LABELS[cls];
  } else {
    switch (cls) {
      case 0: bg = ST77XX_RED;   fg = ST77XX_WHITE; alertText = "AMBULANCE"; break;
      case 1: bg = 0xFD20;       fg = ST77XX_BLACK; alertText = "FIRETRUCK"; break;
      case 2: bg = ST77XX_BLACK; fg = ST77XX_GREEN; alertText = "SAFE";      break;
      case 3: bg = ST77XX_BLUE;  fg = ST77XX_WHITE; alertText = "POLICE";    break;
    }
  }

  tft.fillScreen(bg);

  if (cls == 2 && !thinking) {
    drawCenteredText("SIREN MASTER",  80, 2, ST77XX_GREEN);
    drawCenteredText("SAFE",         150, 4, ST77XX_GREEN);
    drawCenteredText("SCANNING...", 230, 2, ST77XX_GREEN);
  } else if (thinking) {
    drawCenteredText("MENILAI...",  70, 3, fg);
    drawCenteredText(alertText,   150, 3, fg);
    char buf[20];
    snprintf(buf, sizeof(buf), "Conf: %3.0f%%", conf * 100.0f);
    drawCenteredText(buf, 230, 2, fg);
  } else {
    drawCenteredText("WARNING!",   70, 4, fg);
    drawCenteredText(alertText,  160, 4, fg);
    char buf[20];
    snprintf(buf, sizeof(buf), "Conf: %3.0f%%", conf * 100.0f);
    drawCenteredText(buf, 240, 2, fg);
  }
}

// Update HANYA baris confidence (tanpa fillScreen) — CEPAT!
void lcdUpdateConfLine(int cls, float conf, bool thinking) {
  uint16_t bg = ST77XX_BLACK;
  uint16_t fg = ST77XX_WHITE;
  int yConf = 240;

  if (thinking) {
    bg = 0x8200; fg = 0xFD20; yConf = 230;
  } else {
    switch (cls) {
      case 0: bg = ST77XX_RED;  fg = ST77XX_WHITE; break;
      case 1: bg = 0xFD20;     fg = ST77XX_BLACK; break;
      case 3: bg = ST77XX_BLUE; fg = ST77XX_WHITE; break;
    }
  }
  tft.fillRect(0, yConf, 240, 20, bg);
  char buf[20];
  snprintf(buf, sizeof(buf), "Conf: %3.0f%%", conf * 100.0f);
  drawCenteredText(buf, yConf, 2, fg);
}

void lcdShowDetection(int cls, float conf, bool thinking) {
  if (lcdMutex == NULL ||
      xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(30)) != pdTRUE)
    return;

  int stateID = thinking ? (cls + 10) : cls;

  if (stateID != lastShownClass) {
    // State baru → gambar layar penuh SEKALI
    lastShownClass = stateID;
    lastShownConf  = conf;
    strobeToggle   = true;
    lastStrobeUpdate = millis();
    lcdDrawFullState(cls, conf, thinking);
  } else {
    // State sama → update confidence saja (TANPA fillScreen!)
    int oldPct = (int)roundf(lastShownConf * 100.0f);
    int newPct = (int)roundf(conf * 100.0f);
    if (oldPct != newPct) {
      lastShownConf = conf;
      lcdUpdateConfLine(cls, conf, thinking);
    }
  }
  xSemaphoreGive(lcdMutex);
}


// ═══════════════════════════════════════════════════════════════
// STROBE — Non-blocking, hanya ganti warna header bukan fillScreen
// ═══════════════════════════════════════════════════════════════
void updateAlertStrobe() {
  if (!btnManager.isActive()) return;
  // Hanya strobe saat ALERT penuh (bukan SAFE atau THINKING)
  if (lastShownClass < 0 || lastShownClass >= 10) return;
  int cls = lastShownClass; // 0,1,2,3
  if (cls == 2) return;

  unsigned long now = millis();
  if (now - lastStrobeUpdate < 250) return;  // 4Hz strobe (250ms)
  lastStrobeUpdate = now;

  if (lcdMutex == NULL || xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(10)) != pdTRUE) return;

  strobeToggle = !strobeToggle;

  uint16_t bg, fg;
  const char *alertText = "";

  switch (cls) {
    case 0:
      bg = strobeToggle ? ST77XX_RED : ST77XX_BLACK;
      fg = ST77XX_WHITE;
      alertText = "AMBULANCE";
      setRGBSafe(strobeToggle ? 200 : 0, 0, 0);
      break;
    case 1:
      bg = strobeToggle ? 0xFD20 : ST77XX_BLACK;
      fg = strobeToggle ? ST77XX_BLACK : ST77XX_WHITE;
      alertText = "FIRETRUCK";
      setRGBSafe(strobeToggle ? 200 : 0, strobeToggle ? 100 : 0, 0);
      break;
    case 3:
      bg = strobeToggle ? ST77XX_BLUE : ST77XX_BLACK;
      fg = ST77XX_WHITE;
      alertText = "POLICE";
      setRGBSafe(0, 0, strobeToggle ? 200 : 0);
      break;
    default:
      xSemaphoreGive(lcdMutex);
      return;
  }

  // Update HANYA area header + teks (bukan fillScreen!)
  tft.fillRect(0, 0, 240, 60, bg);
  drawCenteredText("WARNING!", 15, 4, fg);

  // Update area label kelas (bukan fillScreen!)
  tft.fillRect(0, 100, 240, 80, bg);
  drawCenteredText(alertText, 120, 4, fg);

  // Conf line sudah diupdate oleh lcdShowDetection, tidak perlu di-redraw
  xSemaphoreGive(lcdMutex);
}


// ═══════════════════════════════════════════════════════════════
// APPLY OUTPUTS — Terapkan RGB + Motor sesuai deteksi
// ═══════════════════════════════════════════════════════════════
void applyOutputs(int cls, float conf, bool thinking) {
  if (!btnManager.isActive()) return;

  if (cls == 2) {
    // SAFE: matikan semua
    setRGBSafe(0, 0, 0);
    motorPattern = 0;
  } else if (thinking) {
    // THINKING: LED solid redup, motor OFF dulu
    switch (cls) {
      case 0: setRGBSafe(100, 0, 0);   break;  // Merah redup
      case 1: setRGBSafe(100, 60, 0);  break;  // Kuning redup
      case 3: setRGBSafe(0, 0, 100);   break;  // Biru redup
    }
    motorPattern = 0;
  } else {
    // ALERT PENUH: Motor aktif, RGB dikendalikan strobe
    switch (cls) {
      case 0: motorPattern = 1; break;  // Ambulance: denyut cepat
      case 1: motorPattern = 2; break;  // Firetruck: 300/200ms
      case 3: motorPattern = 3; break;  // Police: 3x ketuk
    }
  }

  lcdShowDetection(cls, conf, thinking);
}


// ═══════════════════════════════════════════════════════════════
// BUTTON
// ═══════════════════════════════════════════════════════════════
void checkButton() {
  if (btnManager.update()) {
    Serial.println("[POWER] Tombol ditekan! Standby...");
    setRGBSafe(0, 0, 0);
    motorPattern = 0;
    digitalWrite(MOTOR_PIN, LOW);

    if (lcdMutex != NULL) xSemaphoreTake(lcdMutex, portMAX_DELAY);
    tft.fillScreen(ST77XX_BLACK);
    drawCenteredText("SIREN MASTER",  80, 2, 0x4208);
    drawCenteredText("STANDBY",      160, 4, ST77XX_RED);
    drawCenteredText("[ Press to Wake ]", 240, 2, 0x4208);
    delay(1000);
    digitalWrite(TFT_BL, LOW);
    if (lcdMutex != NULL) xSemaphoreGive(lcdMutex);

    btnManager.shutdown(PIN_BUTTON);
  }
}


// ═══════════════════════════════════════════════════════════════
// IO TASK — Button + Motor + Strobe (Core 1, Priority 3)
// ═══════════════════════════════════════════════════════════════
void ioTaskCode(void *pvParameters) {
  for (;;) {
    checkButton();
    updateMotor();
    updateAlertStrobe();
    vTaskDelay(pdMS_TO_TICKS(10));  // 100Hz polling
  }
}


// ═══════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== SirenMaster TinyML (TFLite Micro) ===");

  // GPIO
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);
  btnManager.begin();

  // RGB via LEDC PWM
  setupRGB();
  setRGB(0, 0, 0);

  rgbMutex = xSemaphoreCreateMutex();

  // ── I2S Mic INMP441 ──
  i2s_config_t i2s_config = {
      .mode             = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate      = 16000,
      .bits_per_sample  = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format   = I2S_CHANNEL_FMT_ONLY_LEFT,
#if ESP_ARDUINO_VERSION_MAJOR >= 3
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#else
      .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
#endif
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count    = 8,
      .dma_buf_len      = 512,
      .use_apll         = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk       = 0};
  i2s_pin_config_t pin_config = {
      .bck_io_num   = I2S_SCK_PIN,
      .ws_io_num    = I2S_WS_PIN,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num  = I2S_SD_PIN};
  esp_err_t err1 = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  esp_err_t err2 = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err1 != ESP_OK || err2 != ESP_OK)
    Serial.printf("[ERROR] I2S Gagal! Err1: %d, Err2: %d\n", err1, err2);
  else
    Serial.println("[OK] I2S Mic OK!");

  // ── LCD ST7789 ──
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
  delay(200);
  tft.setSPISpeed(8000000);
  tft.init(240, 320, SPI_MODE3);
  tft.setRotation(2);

  // Splash screen
  tft.fillScreen(ST77XX_BLACK);
  drawCenteredText("SIREN MASTER",   80, 2, ST77XX_CYAN);
  drawCenteredText("SYSTEM STARTING", 125, 1, ST77XX_WHITE);
  tft.drawRect(55, 150, 130, 8, ST77XX_CYAN);
  tft.fillRect(57, 152, 15, 4, ST77XX_CYAN);
  drawCenteredText("Allocating RAM", 175, 1, ST77XX_WHITE);

  // ── Alokasi RAM ──
  tensor_arena  = (uint8_t *)malloc(kTensorArenaSize);
  chunkBuffer   = (int16_t *)malloc(SAMPLE_RATE * sizeof(int16_t));
  historyBuffer = (int16_t *)malloc(AUDIO_LEN   * sizeof(int16_t));
  if (!tensor_arena || !chunkBuffer || !historyBuffer) {
    Serial.println("[ERROR] OUT OF MEMORY!");
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(30, 175);
    tft.println("OUT OF MEMORY!");
    while (1) delay(100);
  }
  memset(historyBuffer, 0, AUDIO_LEN * sizeof(int16_t));

  tft.fillRect(57, 152, 50, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK);
  drawCenteredText("Loading AI Model", 175, 1, ST77XX_WHITE);

  // ── TFLite: Load model ──
  tflModel = tflite::GetModel(siren_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model version mismatch!");
    while (1) delay(1000);
  }

  tft.fillRect(57, 152, 80, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK);
  drawCenteredText("Registering Kernels", 175, 1, ST77XX_WHITE);

  // ── Register ops ──
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddExpandDims();

  tft.fillRect(57, 152, 105, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK);
  drawCenteredText("Allocating Tensors", 175, 1, ST77XX_WHITE);

  // ── Buat interpreter ──
  static tflite::MicroInterpreter static_interpreter(
      tflModel, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("[ERROR] AllocateTensors gagal! Arena=%dKB\n", TENSOR_ARENA_KB);
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(30, 175);
    tft.println("ALLOC ERROR!");
    while (1) delay(1000);
  }

  inputTensor  = interpreter->input(0);
  outputTensor = interpreter->output(0);

  Serial.printf("[OK] Arena %dKB, used %d bytes\n", TENSOR_ARENA_KB,
                interpreter->arena_used_bytes());
  Serial.printf("[IN]  type=%s shape=(%d,%d,%d,%d)\n",
                inputTensor->type == kTfLiteFloat32 ? "FLOAT32" : "INT8",
                inputTensor->dims->data[0], inputTensor->dims->data[1],
                inputTensor->dims->data[2], inputTensor->dims->data[3]);
  Serial.printf("[OUT] type=%s shape=(%d,%d)\n",
                outputTensor->type == kTfLiteFloat32 ? "FLOAT32" : "INT8",
                outputTensor->dims->data[0], outputTensor->dims->data[1]);

  tft.fillRect(57, 152, 126, 4, ST77XX_CYAN);
  tft.fillRect(0, 175, 240, 10, ST77XX_BLACK);
  drawCenteredText("System Ready", 175, 1, ST77XX_WHITE);
  delay(300);

  // Startup test — LED hijau + motor pulse singkat
  setRGBPWM(0, 200, 0);
  digitalWrite(MOTOR_PIN, HIGH);
  tft.fillScreen(ST77XX_BLACK);
  drawCenteredText("SIREN MASTER",  80, 2, ST77XX_GREEN);
  drawCenteredText("READY",        150, 4, ST77XX_GREEN);
  drawCenteredText("SCANNING...", 230, 2, ST77XX_GREEN);
  delay(800);
  digitalWrite(MOTOR_PIN, LOW);
  setRGBPWM(0, 0, 0);
  delay(300);

  motorTimer = millis();

  // ── Start FreeRTOS Tasks ──
  audioSemaphore = xSemaphoreCreateBinary();
  lcdMutex       = xSemaphoreCreateMutex();

  // Core 0: Audio (Priority 2) — Dedicated I2S reading
  BaseType_t res1 = xTaskCreatePinnedToCore(
      audioTaskCode, "TaskAudio", 4096, NULL, 2, &TaskAudio, 0);

  // Core 1: AI Inference (Priority 1) — TFLite
  BaseType_t res2 = xTaskCreatePinnedToCore(
      inferenceTaskCode, "TaskInference", 8192, NULL, 1, &TaskInference, 1);

  // Core 1: IO (Priority 3) — Preempt AI saat button/motor/strobe
  // Stack 4096 cukup, priority 3 > inference agar motor+strobe tidak lag
  BaseType_t res3 = xTaskCreatePinnedToCore(
      ioTaskCode, "TaskIO", 4096, NULL, 3, NULL, 1);

  if (res1 != pdPASS || res2 != pdPASS || res3 != pdPASS) {
    Serial.println("[FATAL] Gagal buat FreeRTOS Task!");
    tft.fillScreen(ST77XX_RED);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(10, 175);
    tft.println("TASK ERROR!");
    while (1) delay(100);
  }
  Serial.println("[OK] Semua task berjalan. Memulai deteksi...");
}


// ═══════════════════════════════════════════════════════════════
// INFERENCE TASK
// ═══════════════════════════════════════════════════════════════
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

void inferenceTaskCode(void *pvParameters) {
  static int  loud_chunks_count = 0;
  static bool is_silent_state   = true;

  for (;;) {
    if (xSemaphoreTake(audioSemaphore, portMAX_DELAY) == pdTRUE) {
      if (!btnManager.isActive()) continue;

      // ── Noise Gate & Clipping Detector ──
      float sumSq = 0;
      int   clip_count = 0;
      for (int i = 0; i < AUDIO_LEN; i++) {
        float val = (float)historyBuffer[i];
        sumSq += val * val;
        if (historyBuffer[i] >= 32760 || historyBuffer[i] <= -32760) clip_count++;
      }
      float rms = sqrtf(sumSq / AUDIO_LEN);

      float threshold_high = noise_floor + 150.0f;
      float threshold_low  = noise_floor + 50.0f;
      bool  check_silent   = is_silent_state ? (rms < threshold_high) : (rms < threshold_low);

      if (check_silent) {
        is_silent_state = true;
        noise_floor = (1.0f - NOISE_FLOOR_ALPHA) * noise_floor + NOISE_FLOOR_ALPHA * rms;
        noise_floor = constrain(noise_floor, NOISE_FLOOR_MIN, NOISE_FLOOR_MAX);
        targetAmplitude = 0.0f;

        // Reset EMA & lock saat hening
        ema_probs[0] = 0.0f; ema_probs[1] = 0.0f;
        ema_probs[2] = 1.0f; ema_probs[3] = 0.0f;
        locked_class_idx  = 2;
        loud_chunks_count = 0;

        bool thinking = false;
        applyOutputs(2, 1.0f, thinking);
        continue;
      }

      is_silent_state = false;
      loud_chunks_count = constrain(loud_chunks_count + 1, 1, 4);
      targetAmplitude   = constrain((rms - noise_floor) * 0.02f, 5.0f, 35.0f);

      extractMelSpec(loud_chunks_count);

      float conf = 0;
      runInference(conf);

      bool  thinking  = false;
      float final_prob = 0.0f;
      int   cls = smartDetectEMA(final_prob, thinking);
      applyOutputs(cls, final_prob, thinking);

      if (cls == 2 && !thinking) {
        noise_floor = (1.0f - NOISE_FLOOR_ALPHA) * noise_floor + NOISE_FLOOR_ALPHA * rms;
        noise_floor = constrain(noise_floor, NOISE_FLOOR_MIN, NOISE_FLOOR_MAX);
      }

      // Serial log
      const char *lbl = getShortLabel(cls, thinking);
      int bar_len = constrain((int)(rms / 300.0f * 6.0f), 0, 6);
      String volBar = "";
      for (int b = 0; b < 6; b++) volBar += (b < bar_len) ? "█" : "░";
      if (clip_count > 20) volBar += " [!] CLIPPING";

      static int spinIdx = 0;
      const char spinner[] = {'|', '/', '-', '\\'};
      char spin = spinner[spinIdx++ % 4];

      Serial.printf("[%c] %s (%.0f%%) | Vol: %s (%.0f)\n\n",
                    spin, lbl, final_prob * 100.0f, volBar.c_str(), rms);
    }
  }
}


// ═══════════════════════════════════════════════════════════════
// LOOP — Semua sudah di FreeRTOS Tasks
// ═══════════════════════════════════════════════════════════════
void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
