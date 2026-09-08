/*
 * sirenmaster_main.ino
 * SirenMaster TinyML — ESP32 WROOM-32
 * Cinematic LCD + Anti-flicker + Dynamic Arena
 */

#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <Fonts/FreeSans9pt7b.h>
#include <Fonts/FreeSansBold12pt7b.h>
#include <Fonts/FreeSansBold18pt7b.h>
#include <SPI.h>
#include <driver/i2s.h>

// ── Pin Definitions ──────────────────────────────────────────────
#define I2S_WS_PIN 5
#define I2S_SCK_PIN 6
#define I2S_SD_PIN 7
#define MOTOR_PIN 13
#define LED_R_PIN 38
#define LED_G_PIN 39
#define LED_B_PIN 40
#define LCD_SCL_PIN 36
#define LCD_SDA_PIN 35
#define LCD_RES_PIN 18
#define LCD_DC_PIN 16
#define LCD_CS_PIN 10
#define PIN_BUTTON                                                             \
  8 // DIUBAH DARI 12 (Pin 12 adalah Strapping Pin yang bikin error upload!)
#define TFT_BLK 9
#define LEDC_FREQ 5000
#define LEDC_RES 8

// ── Audio ─────────────────────────────────────────────────────────
#define SAMPLE_RATE SAMPLE_RATE_HZ
#define AUDIO_LEN AUDIO_SAMPLES
#define FFT_N N_FFT_SIZE
#define FFT_BINS N_FFT_BINS
#define HOP_LEN N_HOP_LENGTH
#define N_FRAMES N_TIME_FRAMES
#define N_MELS N_MEL_FILTERS
// MIC_GAIN 5.5f: Sensitivitas lebih tinggi agar mampu menangkap sirine dari
// speaker laptop maupun dari jarak jauh di jalanan.
#define MIC_GAIN 5.5f

// ── AI ────────────────────────────────────────────────────────────
#define CONFIDENCE_THR 0.50f
#define EMA_ALPHA                                                              \
  0.60f // Dikembalikan ke 0.60: cukup cepat tapi tetap punya efek smoothing
        // anti-bocor
#define SIREN_THRESHOLD 0.88f
#define UNLOCK_THRESHOLD 0.50f
#define THINKING_THRESHOLD 0.80f
#define OVERRIDE_THRESHOLD 0.95f
// Arena = DYNAMIC malloc (bukan static) agar tidak overflow BSS
#define TENSOR_ARENA_KB 96

// RSD (Relative Std Deviation) Filter — dipindah ke global agar bersih
#define RSD_NUM_BLOCKS 32
#define RSD_BLOCK_SIZE 1000

const char *CLASS_LABELS[NUM_CLASSES] = {"AMBULANCE", "FIRETRUCK", "NORMAL",
                                         "POLICE"};

// ── Globals ───────────────────────────────────────────────────────
Adafruit_ST7789 tft = Adafruit_ST7789(LCD_CS_PIN, LCD_DC_PIN, LCD_RES_PIN);

static tflite::MicroMutableOpResolver<14> resolver;
static const tflite::Model *tflModel = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *inputTensor = nullptr;
static TfLiteTensor *outputTensor = nullptr;

// Arena dialokasikan di heap (bukan BSS) → tidak overflow linker
static uint8_t *tensor_arena = nullptr;

static int16_t *chunkBuffer = nullptr;
static int16_t *historyBuffer = nullptr;
static int historyHead = 0;

static float fftReal[FFT_N], fftImag[FFT_N];
static float outputScores[NUM_CLASSES], powerSpec[FFT_BINS];

SemaphoreHandle_t audioSemaphore;
SemaphoreHandle_t lcdMutex;
TaskHandle_t TaskAudio, TaskInference;

static float ema_probs[NUM_CLASSES] = {0, 0, 1, 0};
static int locked_class_idx = 2;
static int weak_siren_streak = 0, last_weak_siren_class = -1;
static float locked_peak_conf = 0.0f; // Skor puncak saat kunci aktif
static unsigned long alert_lock_time = 0;
static int locked_alert_class = 2;
static float locked_alert_conf = 0.0f;

static float dynamic_noise_floor = 200.0f;

static volatile int motorPattern = 0;
static volatile unsigned long motorTimer = 0;
static volatile int motorState = 0;

// ── Fitur Optimal Baru: Snooze Mute, Proximity Meter & Live Timer ─
static volatile unsigned long snoozeUntil = 0;
static unsigned long sirenActiveStartTime = 0;
static volatile float liveVolumeRMS = 0.0f;
static unsigned long lastBtnPressMs = 0;

static int lastLcdClass = -99;
static float lastLcdConf = -1.0f;

// ── Sonar Ping Animation (SYSTEM AMAN) ─────────────────────────────
static volatile int radarCls = 2; // Kelas aktif saat ini (dibaca ioTask)
static volatile unsigned long lastPingMs = 0;
static float pingR[4] = {0.0f, 17.0f, 34.0f,
                         51.0f}; // Radius tiap ring (staggered)

// Backlight brightness (0-255)
static void blSet(uint8_t bright) { ledcWrite(TFT_BLK, bright); }

// Fade in backlight
static void blFadeIn(uint8_t from, uint8_t to, int stepMs) {
  for (int b = from; b <= to; b += 4) {
    blSet(b);
    delay(stepMs);
  }
  blSet(to);
}

// ── RGB (Lampu LED) ────────────────────────────────────────────────
// Ubah ke 'true' jika lampu Anda menyala saat nilai 0 (Common Anode)
// Biarkan 'false' jika lampu Anda normal (Common Cathode)
#define RGB_COMMON_ANODE false

void setupRGB() {
  ledcAttach(LED_R_PIN, LEDC_FREQ, LEDC_RES);
  ledcAttach(LED_G_PIN, LEDC_FREQ, LEDC_RES);
  ledcAttach(LED_B_PIN, LEDC_FREQ, LEDC_RES);
  setRGB(0, 0, 0); // Matikan semua lampu di awal
}

void setRGB(uint8_t r, uint8_t g, uint8_t b) {
  if (RGB_COMMON_ANODE) {
    r = 255 - r;
    g = 255 - g;
    b = 255 - b;
  }
  ledcWrite(LED_R_PIN, r);
  ledcWrite(LED_G_PIN, g);
  ledcWrite(LED_B_PIN, b);
}

// ── LCD Helpers ───────────────────────────────────────────────────
void lcdCenter(const char *text, int y, const GFXfont *font, uint16_t color) {
  tft.setFont(font);
  tft.setTextColor(color);
  tft.setTextSize(1);
  int16_t x1, y1;
  uint16_t w, h;
  tft.getTextBounds(text, 0, 0, &x1, &y1, &w, &h);
  tft.setCursor((tft.width() - w) / 2, y);
  tft.print(text);
}

void drawThickCircle(int cx, int cy, int r, int t, uint16_t c) {
  for (int i = 0; i < t; i++)
    tft.drawCircle(cx, cy, r - i, c);
}

// ── LCD: Animasi Splash Cinematic ────────────────────────────────
void drawSplash() {
  tft.fillScreen(ST77XX_BLACK);
  blSet(0);

  int cx = 140, cy = 90;
  drawThickCircle(cx, cy, 55, 2, 0x2104);

  for (int b = 0; b <= 180; b += 6) {
    blSet(b);
    delay(8);
  }

  for (int r = 15; r <= 45; r += 4) {
    tft.drawCircle(cx, cy, r, 0x07FF);
    delay(10);
  }
  drawThickCircle(cx, cy, 45, 3, 0x07FF);
  tft.fillCircle(cx, cy, 5, 0x07FF);

  delay(100);
  lcdCenter("SIREN", 170, &FreeSansBold18pt7b, ST77XX_WHITE);
  delay(120);
  lcdCenter("MASTER", 210, &FreeSansBold18pt7b, 0x07FF);
  tft.drawFastHLine(80, 225, 120, 0x2945);
  blFadeIn(180, 255, 4);
}

// ── Warna tiap kelas ─────────────────────────────────────────────
uint16_t clsColor(int cls, bool thinking) {
  if (thinking)
    return 0xFD20; // Oranye — Mendeteksi
  switch (cls) {
  case 0:
    return 0xF800; // Merah — Ambulance
  case 1:
    return 0xFC60; // Oranye terang — Firetruck
  case 2:
    return 0x07E0; // Hijau — Aman
  case 3:
    return 0x041F; // Biru — Polisi
  }
  return ST77XX_WHITE;
}

// ── Ikon Ultra Minimal ───────────────────────────────────────────
// AMAN: Checkmark / Centang Minimalis Modern
void drawIconSafe(int cx, int cy, uint16_t c) {
  // Lingkaran luar tebal (4 pixel)
  for (int i = 0; i < 4; i++) {
    tft.drawCircle(cx, cy, 35 - i, c);
  }

  // Tanda centang (Checkmark) yang tebal dan jelas
  int dx = cx - 5;
  int dy = cy + 12;

  // Garis pendek (kiri ke tengah bawah)
  for (int i = -3; i <= 3; i++) {
    tft.drawLine(cx - 18, cy + i, dx, dy + i, c);
  }

  // Garis panjang (tengah bawah ke kanan atas)
  for (int i = -3; i <= 3; i++) {
    tft.drawLine(dx, dy + i, cx + 22, cy - 14 + i, c);
  }
}

// AMBULANCE: Palang Medis besar (+)
void drawIconAmbulance(int cx, int cy, uint16_t c) {
  tft.fillRoundRect(cx - 7, cy - 28, 14, 56, 4, c);
  tft.fillRoundRect(cx - 28, cy - 7, 56, 14, 4, c);
  tft.fillCircle(cx, cy, 5, ST77XX_WHITE);
}

// POLISI: Bintang 5 sudut solid
void drawIconPolice(int cx, int cy, uint16_t c) {
  float r1 = 30.0f, r2 = 13.0f;
  float base = -1.5708f;
  int px[10], py[10];
  for (int i = 0; i < 5; i++) {
    float ao = base + i * 1.2566f;
    float ai = base + i * 1.2566f + 0.6283f;
    px[i * 2] = cx + (int)(r1 * cosf(ao));
    py[i * 2] = cy + (int)(r1 * sinf(ao));
    px[i * 2 + 1] = cx + (int)(r2 * cosf(ai));
    py[i * 2 + 1] = cy + (int)(r2 * sinf(ai));
  }
  for (int i = 0; i < 10; i++) {
    int j = (i + 1) % 10;
    tft.fillTriangle(cx, cy, px[i], py[i], px[j], py[j], c);
  }
}

// FIRETRUCK: Api dua segitiga tumpuk
void drawIconFiretruck(int cx, int cy, uint16_t c) {
  tft.fillTriangle(cx, cy - 32, cx - 22, cy + 22, cx + 22, cy + 22, c);
  tft.fillTriangle(cx, cy - 16, cx - 12, cy + 22, cx + 12, cy + 22, 0xFFE0);
  tft.fillCircle(cx, cy + 10, 6, ST77XX_WHITE);
}

// THINKING: Lingkaran konsentris + sweep
void drawIconThinking(int cx, int cy, uint16_t c) {
  drawThickCircle(cx, cy, 28, 3, c);
  drawThickCircle(cx, cy, 14, 2, c);
  tft.fillCircle(cx, cy, 5, c);
  tft.drawLine(cx, cy, cx + 24, cy - 10, c);
  tft.drawLine(cx, cy, cx + 24, cy - 11, c);
}

// ── Warna Latar / Gelap Tambahan ──────────────────────────────────
uint16_t clsDarkColor(int cls, bool thinking) {
  if (thinking)
    return 0x8200;
  switch (cls) {
  case 0:
    return 0x8000;
  case 1:
    return 0x8200;
  case 2:
    return 0x0400;
  case 3:
    return 0x0100;
  }
  return 0x2104;
}

uint16_t clsDarkerColor(int cls, bool thinking) {
  if (thinking)
    return 0x4100;
  switch (cls) {
  case 0:
    return 0x4000;
  case 1:
    return 0x4100;
  case 2:
    return 0x0200;
  case 3:
    return 0x0008;
  }
  return 0x1082;
}

// ── Smartwatch Floating Pill Card ────────────────────────────────
#define CARD_BG 0x10C3 // Charcoal/Abu-abu sangat gelap

void lcdDrawMonitor(int cls, float conf, bool thinking) {
  lcdUpdateCircle(cls, thinking);
  lcdUpdateLabel(cls, thinking);
  // lcdUpdateConf: dikosongkan, user tidak ingin menampilkan persentase
}

// ── Sonar Ping: dipanggil dari ioTaskCode setiap 10ms ───────────────
void updateSonarPing() {
  // DIMATIKAN SESUAI REQUEST: "jangan memantau dllnya rombak saja yang rapih"
  return;
}

// Update area ikon & bar (Anti-flicker)
void lcdUpdateCircle(int cls, bool thinking) {
  uint16_t cc = clsColor(cls, thinking);
  tft.fillScreen(ST77XX_BLACK);

  // Bingkai siku futuristik (Sci-Fi Corner Brackets) di pojok layar, warna
  // sesuai status
  tft.drawFastHLine(10, 15, 20, cc);
  tft.drawFastVLine(10, 15, 20, cc);
  tft.drawFastHLine(250, 15, 20, cc);
  tft.drawFastVLine(270, 15, 20, cc);
  tft.drawFastHLine(10, 225, 20, cc);
  tft.drawFastVLine(10, 205, 20, cc);
  tft.drawFastHLine(250, 225, 20, cc);
  tft.drawFastVLine(270, 205, 20, cc);

  int cx = 140;
  int cy = 100;

  // === MENGGAMBAR IKON TENGAH ===
  if (thinking)
    drawIconThinking(cx, cy, cc);
  else if (cls == 0)
    drawIconAmbulance(cx, cy, cc);
  else if (cls == 1)
    drawIconFiretruck(cx, cy, cc);
  else if (cls == 2)
    drawIconSafe(cx, cy, cc);
  else if (cls == 3)
    drawIconPolice(cx, cy, cc);
}

void lcdUpdateLabel(int cls, bool thinking) {
  uint16_t cc = clsColor(cls, thinking);
  const char *label;
  switch (cls) {
  case 0:
    label = "AMBULANCE";
    break;
  case 1:
    label = "DAMKAR";
    break;
  case 2:
    label = "AMAN";
    break;
  case 3:
    label = "POLISI";
    break;
  default:
    label = "???";
  }
  if (thinking)
    label = "DETEKSI";

  // Bersihkan area label (tengah bawah)
  tft.fillRect(10, 155, 260, 75, ST77XX_BLACK);

  // 1. Tampilkan Label Utama Kendaraan
  tft.setFont(&FreeSansBold12pt7b);
  tft.setTextColor(cc);
  int16_t x1, y1;
  uint16_t w, h;
  tft.getTextBounds(label, 0, 0, &x1, &y1, &w, &h);
  tft.setCursor((280 - w) / 2, 178);
  tft.print(label);

  if (cls != 2 && !thinking) {
    // Tampilkan label jarak (teks saja, tanpa bar)
    int numBars = 1;
    if (liveVolumeRMS > 3500.0f) numBars = 5;
    else if (liveVolumeRMS > 2600.0f) numBars = 4;
    else if (liveVolumeRMS > 1800.0f) numBars = 3;
    else if (liveVolumeRMS > 1100.0f) numBars = 2;
    else numBars = 1;

    tft.setFont(&FreeSans9pt7b);
    tft.setTextColor(0xAD55); // Abu-abu terang
    const char *distLabel = (numBars >= 5) ? "SANGAT DEKAT!" : ((numBars >= 3) ? "MENDEKAT" : "JARAK JAUH");
    tft.getTextBounds(distLabel, 0, 0, &x1, &y1, &w, &h);
    tft.setCursor((280 - w) / 2, 210);
    tft.print(distLabel);

    // Tampilkan Indikator MUTE jika tombol pernah ditekan
    if (snoozeUntil > millis()) {
      tft.fillRoundRect(180, 18, 90, 18, 3, 0xFFE0); // Kuning terang
      tft.setTextColor(ST77XX_BLACK);
      tft.setCursor(186, 32);
      tft.print("MUTE 10s");
    }
  } else {
    // Status AMAN / SIAGA
    tft.setFont(&FreeSans9pt7b);
    tft.setTextColor(0x07E0); // Hijau lembut
    const char *safeLabel = "SISTEM SIAGA";
    tft.getTextBounds(safeLabel, 0, 0, &x1, &y1, &w, &h);
    tft.setCursor((280 - w) / 2, 208);
    tft.print(safeLabel);
  }
}

// Update persentase secara real-time dengan melokalisir area redraw agar
// zero-flicker
void lcdUpdateConf(int cls, float conf, bool thinking) {
  // Diabaikan/Kosong: User tidak ingin menampilkan persentase dan angka keyakinan
}

void lcdUpdatePills(int cls, bool thinking) {
  // Tidak digunakan
}

// ── Main Show — Tidak fillScreen saat runtime ─────────────────────
void lcdShowState(int cls, float conf, bool thinking) {
  if (!lcdMutex || xSemaphoreTake(lcdMutex, pdMS_TO_TICKS(30)) != pdTRUE)
    return;

  int sid = thinking ? (cls + 10) : cls;

  if (sid != lastLcdClass) {
    lcdUpdateCircle(cls, thinking);
    lcdUpdateLabel(cls, thinking);
  } else if (cls != 2) {
    // Update live proximity bars periodically
    lcdUpdateLabel(cls, thinking);
  }

  lastLcdClass = sid;
  lastLcdConf = conf;
  xSemaphoreGive(lcdMutex);
}

// ── Motor Haptic Patterns (Pola Getar Berbeda per Kendaraan) ──────
// 0: SAFE / MATI
// 1: AMBULANS  -> Double Pulse (Bzz-Bzz ... Bzz-Bzz)
// 2: DAMKAR    -> Heavy Long Pulse (Bzzzzzzz ... Bzzzzzzz)
// 3: POLISI    -> Triple Rapid Staccato (Bz-Bz-Bz ... Bz-Bz-Bz)
void updateMotor() {
  if (motorPattern == 0 || millis() < snoozeUntil) {
    digitalWrite(MOTOR_PIN, LOW);
    motorState = 0;
    return;
  }
  unsigned long now = millis();

  // Pola 1: AMBULANS (Double Pulse: ON 150ms, OFF 100ms, ON 150ms, Jeda 450ms)
  if (motorPattern == 1) {
    static int step1 = 0;
    static unsigned long timer1 = 0;
    const uint32_t iv1[] = {150, 100, 150, 450};
    const uint8_t st1[]  = {1,   0,   1,   0};
    if (now - timer1 >= iv1[step1]) {
      step1 = (step1 + 1) % 4;
      timer1 = now;
      digitalWrite(MOTOR_PIN, st1[step1] ? HIGH : LOW);
    }
  }
  // Pola 2: DAMKAR (Long Heavy Pulse: ON 650ms, Jeda 300ms)
  else if (motorPattern == 2) {
    static int step2 = 0;
    static unsigned long timer2 = 0;
    const uint32_t iv2[] = {650, 300};
    const uint8_t st2[]  = {1,   0};
    if (now - timer2 >= iv2[step2]) {
      step2 = (step2 + 1) % 2;
      timer2 = now;
      digitalWrite(MOTOR_PIN, st2[step2] ? HIGH : LOW);
    }
  }
  // Pola 3: POLISI (Triple Rapid Staccato: ON 80ms, OFF 60ms, ON 80ms, OFF 60ms, ON 80ms, Jeda 400ms)
  else if (motorPattern == 3) {
    static int step3 = 0;
    static unsigned long timer3 = 0;
    const uint32_t iv3[] = {80, 60, 80, 60, 80, 400};
    const uint8_t st3[]  = {1,  0,  1,  0,  1,  0};
    if (now - timer3 >= iv3[step3]) {
      step3 = (step3 + 1) % 6;
      timer3 = now;
      digitalWrite(MOTOR_PIN, st3[step3] ? HIGH : LOW);
    }
  }
}

void applyOutputs(int cls, float conf, bool thinking) {
  // Update radarCls untuk ioTask: hanya AMAN sejati yang boleh jalankan animasi ping
  radarCls = (cls == 2 && !thinking) ? 2 : -1;

  if (cls != 2 && !thinking) {
    if (sirenActiveStartTime == 0) sirenActiveStartTime = millis();
  } else {
    sirenActiveStartTime = 0;
  }

  if (thinking) {
    motorPattern = 0;
    switch (cls) {
    case 0:
      setRGB(40, 0, 0);
      break;
    case 1:
      setRGB(40, 20, 0);
      break;
    case 3:
      setRGB(0, 0, 40);
      break;
    }
  } else if (cls == 2) {
    setRGB(0, 0, 0);
    motorPattern = 0;
  } else {
    switch (cls) {
    case 0:
      setRGB(255, 0, 0);
      motorPattern = 1;
      break;
    case 1:
      setRGB(255, 100, 0);
      motorPattern = 2;
      break;
    case 3:
      setRGB(0, 0, 255);
      motorPattern = 3;
      break;
    }
  }
  lcdShowState(cls, conf, thinking);
}

// ── FFT ───────────────────────────────────────────────────────────
void computeFFT(float *vR, float *vI, int n) {
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = vR[i];
      vR[i] = vR[j];
      vR[j] = t;
      t = vI[i];
      vI[i] = vI[j];
      vI[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * PI / len, wRe = cosf(ang), wIm = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cuRe = 1, cuIm = 0;
      for (int k = 0; k < (len >> 1); k++) {
        float uRe = vR[i + k], uIm = vI[i + k],
              vRe =
                  vR[i + k + (len >> 1)] * cuRe - vI[i + k + (len >> 1)] * cuIm,
              vI2 =
                  vR[i + k + (len >> 1)] * cuIm + vI[i + k + (len >> 1)] * cuRe;
        vR[i + k] = uRe + vRe;
        vI[i + k] = uIm + vI2;
        vR[i + k + (len >> 1)] = uRe - vRe;
        vI[i + k + (len >> 1)] = uIm - vI2;
        float nr = cuRe * wRe - cuIm * wIm;
        cuIm = cuRe * wIm + cuIm * wRe;
        cuRe = nr;
      }
    }
  }
}

// ── Mel Spec ──────────────────────────────────────────────────────
void extractMelSpec(int ac) {
  bool isF = (inputTensor->type == kTfLiteFloat32);
  float *df = isF ? inputTensor->data.f : nullptr;
  int8_t *di = isF ? nullptr : inputTensor->data.int8;
  float sc = isF ? 1.0f : inputTensor->params.scale;
  int zp = isF ? 0 : inputTensor->params.zero_point;
  int as = AUDIO_LEN;

  // 1. Cari rata-rata (DC Offset) untuk menghilangkan bias hardware INMP441
  float dc_offset = 0;
  for (int i = 0; i < as; i++) {
    dc_offset += (float)historyBuffer[i];
  }
  dc_offset /= as;

  // 2. Peak Normalization (100% Identik dengan Python: y = y * min(1.0 / max_val, 10.0))
  // Menyamakan skala volume mikrofon fisik dengan skala file dataset training
  float max_peak = 0.0f;
  for (int i = 0; i < as; i++) {
    float val = fabsf((float)historyBuffer[i] - dc_offset);
    if (val > max_peak)
      max_peak = val;
  }
  float max_val_float = max_peak / 32768.0f;
  float norm_gain = 1.0f;
  if (max_val_float > 1e-6f) {
    norm_gain = min(1.0f / max_val_float, 10.0f); // 100% Identik dengan Python (max 10.0x gain)
  }

  for (int f = 0; f < N_FRAMES; f++) {
    if (f % 10 == 0)
      vTaskDelay(1); // Mencegah Watchdog Timer (WDT) reset / Crash

    int st = f * HOP_LEN;
    float frame_sum = 0;
    for (int k = 0; k < FFT_N; k++) {
      int idx = st + k, tci = (AUDIO_LEN - as) + (idx % as),
          curr = (historyHead + tci) % AUDIO_LEN;

      // 3-Tap Moving Average (Low Pass Filter) identik dengan training
      int prev = (curr - 1 + AUDIO_LEN) % AUDIO_LEN;
      int next = (curr + 1) % AUDIO_LEN;
      float s_prev = (((float)historyBuffer[prev]) - dc_offset) / 32768.0f * norm_gain;
      float s_curr = (((float)historyBuffer[curr]) - dc_offset) / 32768.0f * norm_gain;
      float s_next = (((float)historyBuffer[next]) - dc_offset) / 32768.0f * norm_gain;

      float s = (s_prev + s_curr + s_next) / 3.0f;
      fftReal[k] = s;
      frame_sum += s;
    }

    // [CRITICAL FIX]: Hapus Frame-wise DC Offset persis 100% dengan Python
    float frame_mean = frame_sum / FFT_N;

    for (int k = 0; k < FFT_N; k++) {
      fftReal[k] =
          (fftReal[k] - frame_mean) * pgm_read_float(&HAMMING_WINDOW[k]);
      fftImag[k] = 0;
    }
    computeFFT(fftReal, fftImag, FFT_N);
    for (int k = 0; k < FFT_BINS; k++)
      powerSpec[k] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k];
    for (int m = 0; m < N_MELS; m++) {
      float e = 0;
      for (int k = 0; k < FFT_BINS; k++)
        e += pgm_read_float(&MEL_FILTERBANK[m][k]) * powerSpec[k];
      float feat = (logf(e + 1e-9f) - pgm_read_float(&MEL_MEAN[m])) /
                   pgm_read_float(&MEL_STD[m]);
      int ti = f * N_MELS + m;

      // Keamanan ekstra: jangan tulis di luar batas memori AI
      if (inputTensor != nullptr && ti < (inputTensor->bytes / (isF ? 4 : 1))) {
        if (isF)
          df[ti] = feat;
        else {
          int q = (int)roundf(feat / sc) + zp;
          di[ti] = (int8_t)constrain(q, -128, 127);
        }
      }
    }
  }
}

// ── Inference ─────────────────────────────────────────────────────
int runInference(float &conf) {
  if (!interpreter) {
    conf = 0;
    return 2;
  }
  if (interpreter->Invoke() != kTfLiteOk) {
    conf = 0;
    return 2;
  }
  if (outputTensor->type == kTfLiteInt8) {
    float s = outputTensor->params.scale;
    int z = outputTensor->params.zero_point;
    for (int i = 0; i < NUM_CLASSES; i++)
      outputScores[i] = ((float)outputTensor->data.int8[i] - z) * s;
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
  conf = bestS;
  return (bestS >= CONFIDENCE_THR) ? best : 2;
}

// ── EMA Smart Detect & Stability Filter (Menolak Suara Kipas / Bicara) ──
int smartDetectEMA(float &out, bool &thinking) {
  float cs[NUM_CLASSES], sum = 0;
  for (int i = 0; i < NUM_CLASSES; i++) {
    cs[i] = max(0.0f, outputScores[i]);
    sum += cs[i];
  }
  if (sum > 0)
    for (int i = 0; i < NUM_CLASSES; i++)
      cs[i] /= sum;
  else {
    for (int i = 0; i < NUM_CLASSES; i++)
      cs[i] = (i == 2) ? 1.0f : 0.0f;
  }

  // ── KALIBRASI BIAS ──────────────────────────────────────────────
  // Seimbang 1.0f karena dataset sudah diaudit & dipilah kamarnya 100% konsisten
  const float CAL[NUM_CLASSES] = {1.00f, 1.00f, 1.00f, 1.00f}; // AMB, FIRE, NORM, POL
  float cal_sum = 0;
  for (int i = 0; i < NUM_CLASSES; i++) {
    cs[i] *= CAL[i];
    cal_sum += cs[i];
  }
  if (cal_sum > 0)
    for (int i = 0; i < NUM_CLASSES; i++)
      cs[i] /= cal_sum;

  // Smoothing EMA
  for (int i = 0; i < NUM_CLASSES; i++) {
    ema_probs[i] = (1.0f - EMA_ALPHA) * ema_probs[i] + EMA_ALPHA * cs[i];
  }
  float es = 0;
  for (int i = 0; i < NUM_CLASSES; i++)
    es += ema_probs[i];
  if (es > 0)
    for (int i = 0; i < NUM_CLASSES; i++)
      ema_probs[i] /= es;

  // Cari kelas sirine dengan skor tertinggi saat ini
  int bestSiren = 0;
  float bestSirenScore = cs[0];
  if (cs[1] > bestSirenScore) {
    bestSirenScore = cs[1];
    bestSiren = 1;
  }
  if (cs[3] > bestSirenScore) {
    bestSirenScore = cs[3];
    bestSiren = 3;
  }

  static unsigned long last_siren_time = 0;
  static int current_siren = 2;
  static float locked_peak_conf = 0.0f;
  thinking = false;
  unsigned long now = millis();

  float ema_best_siren = max(ema_probs[0], max(ema_probs[1], ema_probs[3]));

  // ── [TEMPORAL CONSENSUS HARD SIREN LOCKING] ─────────────────────
  // Total akumulasi probabilitas sirine (Amb + Fire + Pol)
  float totalSiren = cs[0] + cs[1] + cs[3];
  float ema_totalSiren = ema_probs[0] + ema_probs[1] + ema_probs[3];

  // 1. Jika saat ini sedang mengunci sirine valid (AMBULANCE / DAMKAR / POLISI):
  // KUNCI KELAS MUTLAK DITAHAN 100% DARI AWAL SAMPAI AKHIR PEMUTARAN!
  // Dilarang keras melompat/berubah ke kelas sirine lain di tengah-tengah lagu.
  if (current_siren != 2) {
    // Jika hening / Normal murni mendominasi kuat (Normal >= 70% dan jeda > 2.5 detik):
    if (cs[2] >= 0.70f && (now - last_siren_time > 2500UL)) {
      current_siren = 2;
      locked_peak_conf = 0.0f;
      out = cs[2];
      return 2;
    }

    // Selama sirine masih terdengar (total sirine / nada aktif):
    if (totalSiren >= 0.35f || cs[current_siren] >= 0.25f || (now - last_siren_time < 3000UL)) {
      if (cs[current_siren] > 0.20f || totalSiren > 0.35f) {
        last_siren_time = now;
      }
      out = max(locked_peak_conf, max(cs[current_siren], ema_probs[current_siren]));
      return current_siren; // PERTAHANKAN KUNCI SIRINE DARI AWAL SAMPAI AKHIR!
    }

    // Jika hening berlanjut > 3 detik, baru kembalikan ke SAFE
    current_siren = 2;
    locked_peak_conf = 0.0f;
    out = cs[2];
    return 2;
  }

  // 2. Jika status awal saat ini SAFE (belum ada sirine mengunci):
  // Pemicu Awal Peka & Respon Cepat (Putaran 1-2 langsung mengunci seketika):
  if ((totalSiren >= 0.30f && bestSirenScore >= 0.20f && bestSirenScore > max(cs[2] * 0.30f, 0.15f)) ||
      (bestSirenScore >= 0.30f && bestSirenScore > cs[2])) {
    current_siren = bestSiren;
    locked_peak_conf = max(bestSirenScore, ema_best_siren);
    last_siren_time = now;
    out = locked_peak_conf;
    return current_siren;
  }

  // 3. Kembali ke status SAFE jika tidak ada sirine aktif
  current_siren = 2;
  locked_peak_conf = 0.0f;
  out = cs[2];
  return 2;
}

// ── Tasks ─────────────────────────────────────────────────────────
void audioTaskCode(void *p) {
  int32_t *rb = (int32_t *)malloc(512 * 4);
  int ci = 0;
  for (;;) {
    while (ci < SAMPLE_RATE) {
      size_t bi = 0;
      esp_err_t err = i2s_read(I2S_NUM_0, rb, 512 * 4, &bi, pdMS_TO_TICKS(100));

      if (err != ESP_OK || bi == 0) {
        Serial.println("[ERR] Mic INMP441 tidak merespon! Cek kabel WS/SCK/SD.");
        vTaskDelay(pdMS_TO_TICKS(500));
        continue;
      }

      int n = bi / 4;
      for (int j = 0; j < n; j++) {
        if (ci >= SAMPLE_RATE)
          break;
        int32_t sample = rb[j];
        int16_t s = (int16_t)(sample >> 14);
        chunkBuffer[ci++] = s;
      }
    }
    ci = 0;
    for (int i = 0; i < SAMPLE_RATE; i++) {
      historyBuffer[historyHead] = chunkBuffer[i];
      historyHead = (historyHead + 1) % AUDIO_LEN;
    }
    xSemaphoreGive(audioSemaphore);
  }
}

void inferenceTaskCode(void *p) {
  static int lc = 0;
  for (;;) {
    if (xSemaphoreTake(audioSemaphore, portMAX_DELAY) != pdTRUE)
      continue;

    float dc_sum = 0;
    for (int i = 0; i < AUDIO_LEN; i++)
      dc_sum += (float)historyBuffer[i];
    float dc = dc_sum / AUDIO_LEN;

    float sq = 0;
    for (int i = 0; i < AUDIO_LEN; i++) {
      float v = (float)historyBuffer[i] - dc;
      sq += v * v;
    }
    float rms_raw = sqrtf(sq / AUDIO_LEN);
    float rms = rms_raw * MIC_GAIN;
    liveVolumeRMS = rms;

    if (rms < dynamic_noise_floor) {
      dynamic_noise_floor = 0.95f * dynamic_noise_floor + 0.05f * rms;
    } else {
      dynamic_noise_floor = 0.999f * dynamic_noise_floor + 0.001f * rms;
    }
    if (dynamic_noise_floor < 120.0f)
      dynamic_noise_floor = 120.0f;
    if (dynamic_noise_floor > 800.0f)
      dynamic_noise_floor = 800.0f;

    // Gerbang Volume Dinamis & Filter RSD:
    // Diturunkan ke 450 agar sangat sensitif terhadap sirine jauh/speaker laptop
    float volume_gate = max(dynamic_noise_floor * 1.40f, 450.0f);
    static int quiet_streak = 0;

    // Analisis Kontinuitas Amplitudo (RSD Filter):
    float rms_blocks[RSD_NUM_BLOCKS];
    float rms_sum = 0;
    for (int b = 0; b < RSD_NUM_BLOCKS; b++) {
      float block_sq = 0;
      for (int i = 0; i < RSD_BLOCK_SIZE; i++) {
        int idx = b * RSD_BLOCK_SIZE + i;
        float v = (float)historyBuffer[idx] - dc;
        block_sq += v * v;
      }
      rms_blocks[b] = sqrtf(block_sq / RSD_BLOCK_SIZE);
      rms_sum += rms_blocks[b];
    }
    float rms_mean = rms_sum / RSD_NUM_BLOCKS;
    float rms_var = 0;
    for (int b = 0; b < RSD_NUM_BLOCKS; b++) {
      float diff = rms_blocks[b] - rms_mean;
      rms_var += diff * diff;
    }
    float rms_std = sqrtf(rms_var / RSD_NUM_BLOCKS);
    float rsd = rms_std / (rms_mean + 1e-9f);

    bool is_quiet = (rms < volume_gate);
    bool is_non_continuous = (rsd > 1.60f && rms < 3000.0f);

    if (is_quiet || is_non_continuous) {
      quiet_streak++;
      // [ANTI-FLICKER HYSTERESIS]:
      // Jika sebelumnya SAFE (2), langsung respon aman.
      // Tapi jika SIRENE AKTIF, tunggu minimal 6 frame hening berturut-turut (~3 detik)
      // agar tidak kaget/putus saat ada jeda ayunan wail atau fluktuasi speaker!
      if (lastLcdClass == 2 || quiet_streak >= 6) {
        lc = 0;
        bool th = false;
        applyOutputs(2, 1.0f, th);
        if (lastLcdClass != 2) {
          Serial.printf("[SAFE] Sirine Selesai (Sunyi %d frame berturut-turut)\n", quiet_streak);
        }
      }
      continue;
    } else {
      quiet_streak = 0;
    }

    // Ada suara kontinu (potensial sirene) — jalankan AI
    lc = constrain(lc + 1, 1, 4);
    extractMelSpec(lc);
    float conf = 0;
    runInference(conf);
    bool thinking = false;
    float fp = 0;
    int cls = smartDetectEMA(fp, thinking);
    applyOutputs(cls, fp, thinking);

    if (cls == 2) {
      Serial.printf("[SAFE] Vol: %.0f | Amb: %.0f%% | Fire: %.0f%% | Norm: "
                    "%.0f%% | Pol: %.0f%%\n",
                    rms, outputScores[0] * 100.0f, outputScores[1] * 100.0f,
                    outputScores[2] * 100.0f, outputScores[3] * 100.0f);
    } else {
      Serial.printf("[%s] SIRENE! conf=%.2f (Vol: %.0f) | Amb: %.0f%% | Fire: %.0f%% | Norm: %.0f%% | Pol: %.0f%%\n",
                    CLASS_LABELS[cls], fp, rms, outputScores[0] * 100.0f,
                    outputScores[1] * 100.0f, outputScores[2] * 100.0f, outputScores[3] * 100.0f);
    }
  }
}

void ioTaskCode(void *p) {
  for (;;) {
    // ── Tombol Fisik (GPIO 8): Mute / Snooze 10 Detik ──────────────
    if (digitalRead(PIN_BUTTON) == LOW) {
      unsigned long now = millis();
      if (now - lastBtnPressMs > 350) {
        lastBtnPressMs = now;
        if (snoozeUntil > now) {
          snoozeUntil = 0;
          Serial.println("[BUTTON] SNOOZE DIMATIKAN -> Getaran Normal Kembali");
        } else {
          snoozeUntil = now + 10000;
          Serial.println("[BUTTON] SNOOZE 10 DETIK DIAKTIFKAN -> Getaran Disenyapkan");
        }
      }
    }

    updateMotor();
    updateSonarPing(); // Animasi sonar ping real-time saat SYSTEM AMAN
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// ── SETUP ─────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("==============================");
  Serial.println("=== SirenMaster ESP32 WROOM ===");
  Serial.println("==============================");

  // ── TOMBOL MUTE & MOTOR SETUP ────────────────────────────────────
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  Serial.printf("[BUTTON] GPIO%d = INPUT_PULLUP (Snooze Mute 10s)\n", PIN_BUTTON);

  // Pindahkan setup RGB ke paling atas agar bisa berkedip bersamaan dengan Motor
  setupRGB();

  pinMode(MOTOR_PIN, OUTPUT);
  Serial.printf("[MOTOR] GPIO%d = OUTPUT\n", MOTOR_PIN);

  // Test 1: ON penuh 600ms (Getar Panjang + RGB Putih)
  Serial.println("[MOTOR+RGB] TEST ON 600ms...");
  digitalWrite(MOTOR_PIN, HIGH);
  setRGB(255, 255, 255); // RGB Nyala Putih Terang
  delay(600);
  digitalWrite(MOTOR_PIN, LOW);
  setRGB(0, 0, 0); // RGB Mati
  delay(300);

  // Test 2: 3x pulse getar + RGB Kelap-kelip Merah, Hijau, Biru
  Serial.println("[MOTOR+RGB] Pulse 3x...");
  for (int p = 0; p < 3; p++) {
    digitalWrite(MOTOR_PIN, HIGH);
    if (p == 0)
      setRGB(255, 0, 0); // Merah
    if (p == 1)
      setRGB(0, 255, 0); // Hijau
    if (p == 2)
      setRGB(0, 0, 255); // Biru
    Serial.printf("[MOTOR] pulse %d ON\n", p + 1);
    delay(100);
    digitalWrite(MOTOR_PIN, LOW);
    setRGB(0, 0, 0);
    delay(100);
  }
  Serial.println("[MOTOR] Test selesai.");
  delay(200);

  // Backlight via LEDC (untuk animasi fade)
  ledcAttach(TFT_BLK, 1000, 8);
  blSet(0); // Gelap dulu

  // ── I2S Mic ───────────────────────────────────────────────────────
  i2s_config_t ic = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format =
          (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 128,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};
  i2s_pin_config_t pc = {.bck_io_num = I2S_SCK_PIN,
                         .ws_io_num = I2S_WS_PIN,
                         .data_out_num = I2S_PIN_NO_CHANGE,
                         .data_in_num = I2S_SD_PIN};
  esp_err_t err1 = i2s_driver_install(I2S_NUM_0, &ic, 0, NULL);
  esp_err_t err2 = i2s_set_pin(I2S_NUM_0, &pc);

  if (err1 != ESP_OK || err2 != ESP_OK) {
    Serial.println("[FATAL] GAGAL Menginstal Driver I2S Mic!");
  } else {
    Serial.println("[OK] I2S Mic berhasil dikonfigurasi.");
  }
  // LCD
  Serial.println("[DEBUG] Mengaktifkan Backlight...");
  blSet(255); // Nyalakan backlight agar layar langsung terlihat

  Serial.println("[DEBUG] Inisialisasi SPI LCD (SCL=36, SDA=35, CS=10)...");
  SPI.begin(LCD_SCL_PIN, -1, LCD_SDA_PIN, LCD_CS_PIN);
  delay(100);

  Serial.println("[DEBUG] Inisialisasi ST7789 Display...");
  tft.init(240, 280);
  tft.setRotation(3); // Mode Landscape
  tft.fillScreen(ST77XX_BLACK);
  Serial.println("[OK] Layar ST7789 Terdeteksi & Berhasil Diinisialisasi!");

  // ── CINEMATIC SPLASH (Tampilan Booting) ─────────────────────────
  Serial.println("[DEBUG] Menampilkan Booting Splash Screen...");
  drawSplash();
  delay(800);

  // Buffer
  Serial.println("[DEBUG] Pre malloc...");
  historyBuffer = (int16_t *)malloc(AUDIO_LEN * sizeof(int16_t));
  chunkBuffer = (int16_t *)malloc(SAMPLE_RATE * sizeof(int16_t));
  tensor_arena = (uint8_t *)malloc(TENSOR_ARENA_KB * 1024);
  if (!historyBuffer || !chunkBuffer || !tensor_arena) {
    Serial.println("[FATAL] malloc gagal!");
    while (1)
      delay(100);
  }
  memset(historyBuffer, 0, AUDIO_LEN * sizeof(int16_t));
  Serial.printf("[RAM] Free: %d B\n", ESP.getFreeHeap());
  Serial.println("[DEBUG] malloc OK");

  // Progress ring sederhana: 3 titik loading (Geser Y ke 220 untuk Landscape)
  tft.fillRect(0, 215, 280, 10, ST77XX_BLACK);
  int16_t x1, y1;
  uint16_t w, h;
  // Dots loading animation
  for (int d = 0; d < 3; d++) {
    tft.fillCircle(120 + d * 20, 220, 3, 0x07FF);
    delay(200);
  }

  // Load model
  Serial.println("[DEBUG] Pre GetModel...");
  tflModel = tflite::GetModel(siren_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERR] Model!");
    while (1)
      delay(100);
  }

  // Loading dots
  for (int d = 0; d < 3; d++) {
    tft.fillCircle(120 + d * 20, 220, 3, d < 2 ? 0x2945 : 0x07FF);
    tft.fillCircle(120 + d * 20, 220, 3, 0x07FF);
    delay(100);
  }

  // Ops
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddPad();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddRelu();
  resolver.AddAveragePool2D();

  Serial.println("[DEBUG] Pre MicroInterpreter...");
  static tflite::MicroInterpreter si(tflModel, resolver, tensor_arena,
                                     TENSOR_ARENA_KB * 1024);
  interpreter = &si;
  Serial.println("[DEBUG] MicroInterpreter OK");

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERR] AllocateTensors gagal!");
    // Tidak tampilkan di LCD, tidak restart — log saja
    // Tetap lanjut tapi inference tidak berjalan (LCD tetap monitoring)
  } else {
    inputTensor = interpreter->input(0);
    outputTensor = interpreter->output(0);
    Serial.println("[OK] AI siap!");
  }

  // ── Transisi: Fade out splash, fade in monitoring ─────────────────
  // Fade out
  for (int b = 255; b >= 0; b -= 8) {
    blSet(b);
    delay(5);
  }

  // Gambar layar monitoring (SAFE awal) — layar masih gelap
  lcdDrawMonitor(2, 1.0f, false);
  lastLcdClass = 2;
  lastLcdConf = 1.0f;

  // Fade in monitoring
  blFadeIn(0, 255, 5);

  motorTimer = millis();

  // Tasks
  audioSemaphore = xSemaphoreCreateBinary();
  lcdMutex = xSemaphoreCreateMutex();

  xTaskCreatePinnedToCore(audioTaskCode, "TaskAudio", 8192, NULL, 2, &TaskAudio,
                          1);
  xTaskCreatePinnedToCore(inferenceTaskCode, "TaskInference", 16384, NULL, 1,
                          &TaskInference, 0);
  xTaskCreatePinnedToCore(ioTaskCode, "TaskIO", 2048, NULL, 3, NULL, 1);

  Serial.println("[OK] SirenMaster aktif!");
}

void loop() { vTaskDelay(portMAX_DELAY); }
