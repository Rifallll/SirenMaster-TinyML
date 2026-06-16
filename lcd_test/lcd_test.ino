/*
 * LCD TEST - Tes Layar ST7789 Saja
 */

#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>

#define LCD_SCL_PIN  18
#define LCD_SDA_PIN  23
#define LCD_RES_PIN  4
#define LCD_DC_PIN   2
#define LCD_CS_PIN   5
#define TFT_BL       15

Adafruit_ST7789 tft = Adafruit_ST7789(LCD_CS_PIN, LCD_DC_PIN, LCD_RES_PIN);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=== LCD TEST START ===");

  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
  Serial.println("[1] Backlight ON");

  Serial.println("[2] Mulai tft.init()...");
  tft.init(240, 240);
  Serial.println("[3] tft.init() SELESAI!");

  tft.setRotation(2);
  tft.fillScreen(ST77XX_RED);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(3);
  tft.setCursor(50, 100);
  tft.println("HELLO!");
  Serial.println("[4] HELLO tertulis!");
  Serial.println("=== LCD TEST DONE ===");
}

void loop() {
  delay(2000);
  tft.fillScreen(ST77XX_RED);
  tft.setCursor(50, 100);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(3);
  tft.println("MERAH");

  delay(2000);
  tft.fillScreen(ST77XX_BLUE);
  tft.setCursor(50, 100);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(3);
  tft.println("BIRU");
}
