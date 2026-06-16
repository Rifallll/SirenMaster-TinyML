#include "buttonmanager.h"
#include <esp_sleep.h>

ButtonManager::ButtonManager(uint8_t buttonPin) {
    pin = buttonPin;
    lastState = HIGH;
    lastDebounce = 0;
    systemState = true;
}

void ButtonManager::begin() {
    pinMode(pin, INPUT_PULLUP);
}

bool ButtonManager::update() {
    bool current = digitalRead(pin);

    // Tombol ditekan (LOW) dan sebelumnya dilepas (HIGH)
    if (current == LOW && lastState == HIGH) {
        if (millis() - lastDebounce > 300) {
            systemState = !systemState;
            lastDebounce = millis();
            lastState = current;
            Serial.printf("[BUTTON] System %s\n", systemState ? "ON" : "OFF");
            return true;
        }
    }

    // Reset saat tombol dilepas (dengan cooldown)
    if (current == HIGH && (millis() - lastDebounce > 300)) {
        lastState = HIGH;
    }

    return false;
}

bool ButtonManager::isActive() {
    return systemState;
}

void ButtonManager::shutdown(uint8_t wakeupPin) {
    Serial.println("[POWER] ESP32 entering DEEP SLEEP...");
    Serial.flush();
    delay(100);
    
    // Konfigurasi pin wakeup: bangun saat tombol ditekan (LOW)
    esp_sleep_enable_ext0_wakeup((gpio_num_t)wakeupPin, 0);
    
    // MATIKAN ESP32 TOTAL
    esp_deep_sleep_start();
    // --- Kode di bawah ini TIDAK AKAN PERNAH dieksekusi ---
}