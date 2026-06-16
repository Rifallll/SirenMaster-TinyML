#include "ButtonManager.h"

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

    if (current == LOW &&
        lastState == HIGH &&
        millis() - lastDebounce > 300) {

        systemState = !systemState;
        lastDebounce = millis();

        Serial.printf(
            "[BUTTON] System %s\n",
            systemState ? "ON" : "OFF"
        );

        lastState = current;
        return true;
    }

    lastState = current;
    return false;
}

bool ButtonManager::isActive() {
    return systemState;
}