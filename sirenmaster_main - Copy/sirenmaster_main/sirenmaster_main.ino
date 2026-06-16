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
#define LCD_DC_PIN   2   // Data/Command (Kabel Ungu ke D2)
#define LCD_CS_PIN   -1   // 7-Pin ST7789 tidak pakai CS, set -1

// Others
#define PIN_BUTTON   25   // Moved from 33 (now used by I2S_SCK)
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
#define CONFIDENCE_THR 0.50f
#define TENSOR_ARENA_KB 85
#define SMOOTH_COUNT 3

const char *CLASS_LABELS[NUM_CLASSES] = {"AMBULANCE", "FIRETRUCK", "NOISE",
                                         "POLICE"};

// ═══════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════
Adafruit_ST7789 tft = Adafruit_ST7789(LCD_CS_PIN, LCD_DC_PIN, LCD_SDA_PIN, LCD_SCL_PIN, LCD_RES_PIN);

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
static bool systemActive = true;
static bool lastBtnState = HIGH;
static unsigned long btnDebounce = 0;

// Smart detect
static int lastDetectedClass = 2, consecutiveCount = 0, confirmedClass = 2;

// ═══════════════════════════════════════════════════════════════
// AUDIO
// ═══════════════════════════════════════════════════════════════
void audioTaskCode(void *pvParameters) {
  const int chunk_size = SAMPLE_RATE; // 1 detik = 8000 sampel
  size_t bytesIn = 0;
  int32_t sample32 = 0;

  for(;;) {
    if (!systemActive) {
      delay(100);
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
    for (int k = 0; k < FFT_N; k++) {
      int curr = (historyHead + start + k) % AUDIO_LEN;
      int prev = (curr == 0) ? (AUDIO_LEN - 1) : (curr - 1);
      int next = (curr + 1) % AUDIO_LEN;
      
      // On-the-fly Low Pass Filter
      float sample = (historyBuffer[prev] + historyBuffer[curr] + historyBuffer[next]) / (3.0f * 32768.0f);
      fftReal[k] = sample * pgm_read_float(&HAMMING_WINDOW[k]);
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

#define HISTORY_LEN 3
static int classHistory[HISTORY_LEN] = {2, 2, 2};
static int historyIdx = 0;
static unsigned long lastSirenTime = 0;
static int heldClass = 2;

int smartDetect(int raw) {
  unsigned long now = millis();
  
  // 1. Simpan tebakan terbaru ke dalam riwayat
  classHistory[historyIdx] = raw;
  historyIdx = (historyIdx + 1) % HISTORY_LEN;
  
  // 2. Lakukan Voting (Hitung Suara Terbanyak dalam 3 detik terakhir)
  int votes[NUM_CLASSES] = {0};
  for(int i=0; i<HISTORY_LEN; i++) {
    votes[classHistory[i]]++;
  }
  
  // Cari Pemenang Voting
  int winner = 2;
  int maxVotes = 0;
  for(int i=0; i<NUM_CLASSES; i++) {
    if(votes[i] > maxVotes) {
      maxVotes = votes[i];
      winner = i;
    }
  }
  
  // Keputusan akhir: Harus menang telak (minimal 2 dari 3 suara), jika seri/acak, ikuti yang terakhir
  int decision = (maxVotes >= 2) ? winner : raw;
  
  if (decision != 2) { // JIKA ADA SIRINE (Bukan NORMAL)
    heldClass = decision;
    lastSirenTime = now;
    return heldClass;
  } else { // JIKA NORMAL (Hening)
    // Tahan status sirine terakhir selama 2 detik agar tidak berkedip
    if (now - lastSirenTime < 2000 && heldClass != 2) {
      return heldClass;
    } else {
      heldClass = 2;
      return 2;
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// OUTPUT (LED, Motor, LCD)
// ═══════════════════════════════════════════════════════════════
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

// Static tracking variables for flicker-free display updates
static int lastShownClass = -1;
static float lastShownConf = -1.0f;

void lcdShowDetection(int cls, float conf) {
  if (cls != lastShownClass) {
    uint16_t bg = ST77XX_BLACK;
    uint16_t fg = ST77XX_WHITE;
    const char *alertText = "";

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

    tft.fillScreen(bg);

    // Draw smartwatch style outer ring/border
    tft.drawRoundRect(10, 10, 220, 260, 20, fg);
    tft.drawRoundRect(11, 11, 218, 258, 19, fg);

    // Title / Header
    tft.setTextSize(2);
    tft.setTextColor(fg);
    int titleW =
        (cls == 2)
            ? (9 * 12)
            : (13 * 12); // "Listening" is 9 chars, "!! WARNING !!" is 13 chars
    if (cls == 2) {
      tft.setCursor((240 - titleW) / 2, 50);
      tft.print("Listening");
    } else {
      tft.setCursor((240 - titleW) / 2, 50);
      tft.print("!! WARNING !!");
    }

    // Main Status Text
    tft.setTextSize(3);
    int textW = strlen(alertText) * 18; // 18 pixels width per char at size 3
    tft.setCursor((240 - textW) / 2, 110);
    tft.print(alertText);

    // Confidence Label
    tft.setTextSize(2);
    tft.setCursor(45, 170);
    tft.print("Conf:");

    lastShownClass = cls;
    lastShownConf = -1.0f; // Force redraw of dynamic elements
  }

  // Update dynamic elements (Confidence & Bar)
  if (fabs(conf - lastShownConf) > 0.01f) {
    uint16_t bg = ST77XX_BLACK;
    uint16_t fg = ST77XX_WHITE;
    if (cls == 0) {
      bg = ST77XX_RED;
      fg = ST77XX_WHITE;
    } else if (cls == 1) {
      bg = 0xFD20;
      fg = ST77XX_BLACK;
    } else if (cls == 3) {
      bg = ST77XX_BLUE;
      fg = ST77XX_WHITE;
    }

    // Clear and redraw percentage
    tft.fillRect(115, 170, 80, 20, bg);
    tft.setTextSize(2);
    tft.setTextColor(fg);
    tft.setCursor(115, 170);
    tft.printf("%3.0f%%", conf * 100);

    // Progress Bar
    int maxW = 150;
    int barW = (int)(conf * maxW);
    if (barW > maxW)
      barW = maxW;
    if (barW < 0)
      barW = 0;

    tft.drawRect(45, 210, maxW, 16, fg);
    if (barW > 0)
      tft.fillRect(45, 210, barW, 16, fg);
    if (maxW - barW > 0)
      tft.fillRect(45 + barW, 210, maxW - barW, 16, bg);

    lastShownConf = conf;
  }
}

void applyOutputs(int cls, float conf) {
  switch (cls) {
  case 0:
    setRGB(1, 0, 0);
    motorPattern = 1;
    break;
  case 1:
    setRGB(1, 1, 0);
    motorPattern = 2;
    break;
  case 2:
    setRGB(0, 0, 0);
    motorPattern = 0;
    break;
  case 3:
    setRGB(0, 0, 1);
    motorPattern = 3;
    break;
  }
  // Use the smoothed class's probability as the displayed confidence level
  float smoothedConf = outputScores[cls];
  lcdShowDetection(cls, smoothedConf);
}

// ═══════════════════════════════════════════════════════════════
// BUTTON
// ═══════════════════════════════════════════════════════════════
void checkButton() {
  bool btn = digitalRead(PIN_BUTTON);
  if (btn == LOW && lastBtnState == HIGH && (millis() - btnDebounce > 300)) {
    systemActive = !systemActive;
    btnDebounce = millis();
    Serial.printf("[BTN] System %s\n", systemActive ? "ON" : "OFF");
    if (!systemActive) {
      setRGB(0, 0, 0);
      motorPattern = 0;
      digitalWrite(MOTOR_PIN, LOW);
      tft.fillScreen(ST77XX_BLACK);
      tft.setTextColor(ST77XX_YELLOW);
      tft.setTextSize(2);
      tft.setCursor(30, 120);
      tft.println("PAUSED");
    }
  }
  lastBtnState = btn;
}

// ═══════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== TinyML Siren Detector (TFLite Micro) ===");

  pinMode(LED_R_PIN, OUTPUT);
  pinMode(LED_G_PIN, OUTPUT);
  pinMode(LED_B_PIN, OUTPUT);
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
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
  tft.init(240, 240); // Resolusi layar kotak 1.3 inch
  tft.setRotation(2); // Putar layar agar tulisan tidak terbalik/miring
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_CYAN);
  tft.setTextSize(2);
  tft.setCursor(10, 80);
  tft.println("Siren Detector");
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(10, 120);
  tft.println("Loading TFLite model...");

  // ── ALOKASI MEMORI MURNI RAM INTERNAL (WROOM SAFE) ──
  tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  chunkBuffer = (int16_t*)malloc(SAMPLE_RATE * sizeof(int16_t));
  historyBuffer = (int16_t*)malloc(AUDIO_LEN * sizeof(int16_t));

  if (!tensor_arena || !chunkBuffer || !historyBuffer) {
    Serial.println("[ERROR] GAGAL Mengalokasikan Memori! Alat berhenti.");
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(10, 140);
    tft.println("OUT OF MEMORY!");
    while(1) delay(100);
  }
  memset(historyBuffer, 0, AUDIO_LEN * sizeof(int16_t));

  // ── TFLite: Load model ──
  tflModel = tflite::GetModel(siren_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[ERROR] Model version mismatch!");
    while (1)
      delay(1000);
  }

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

  // ── Buat interpreter ──
  static tflite::MicroInterpreter static_interpreter(
      tflModel, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("[ERROR] AllocateTensors gagal! Arena=%dKB\n",
                  TENSOR_ARENA_KB);
    tft.setTextColor(ST77XX_RED);
    tft.setCursor(10, 140);
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

  // Startup test
  setRGB(1, 0, 0);
  delay(200);
  setRGB(0, 1, 0);
  delay(200);
  setRGB(0, 0, 1);
  delay(200);
  setRGB(0, 0, 0);
  digitalWrite(MOTOR_PIN, HIGH);
  delay(200);
  digitalWrite(MOTOR_PIN, LOW);

  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_GREEN);
  tft.setTextSize(2);
  tft.setCursor(20, 120);
  tft.println("READY");
  Serial.println("[Setup] Done\n");
  motorTimer = millis();
  
  // Start FreeRTOS Tasks
  audioSemaphore = xSemaphoreCreateBinary();
  BaseType_t res1 = xTaskCreatePinnedToCore(
    audioTaskCode, "TaskAudio", 4096, NULL, 2, &TaskAudio, 0); // Core 0 untuk I2S
    
  BaseType_t res2 = xTaskCreatePinnedToCore(
    inferenceTaskCode, "TaskInference", 8192, NULL, 1, &TaskInference, 1); // Core 1 untuk AI

  if (res1 != pdPASS || res2 != pdPASS) {
    Serial.println("[FATAL ERROR] Gagal membuat FreeRTOS Task! Kehabisan Memori RAM.");
    tft.fillScreen(ST77XX_RED);
    tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(10, 140);
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
      if (!systemActive) continue;
      
      // --- NOISE GATE ---
      float sumSq = 0;
      for(int i = 0; i < AUDIO_LEN; i++) {
        sumSq += ((float)historyBuffer[i] * historyBuffer[i]);
      }
      float rms = sqrt(sumSq / AUDIO_LEN);
      
      if (rms < 80.0f) {
        Serial.printf("[ZzZ] Hening (RMS: %.1f) -> SAFE\n\n", rms);
        int cls = smartDetect(2);
        applyOutputs(cls, 1.0f);
        continue;
      }

      Serial.println("[DSP] Mel-Spectrogram...");
      extractMelSpec();

      Serial.println("[ML] Inference...");
      float conf = 0;
      int raw = runInference(conf);
      int cls = smartDetect(raw);
      applyOutputs(cls, conf);

      Serial.printf("[Result] %s | %.1f%%\n\n", CLASS_LABELS[cls], conf * 100);
    }
  }
}

void loop() {
  // Loop is now just for button & motor updates (Fast polling)
  checkButton();
  updateMotor();
  delay(10);
}
