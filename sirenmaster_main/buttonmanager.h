#ifndef BUTTON_MANAGER_H
#define BUTTON_MANAGER_H

#include <Arduino.h>

class ButtonManager {
private:
    uint8_t pin;
    bool lastState;
    unsigned long lastDebounce;
    bool systemState;

public:
    ButtonManager(uint8_t buttonPin);

    void begin();
    bool update();
    bool isActive();
    void shutdown(uint8_t wakeupPin);
};

#endif