#include <Adafruit_NeoPixel.h>
#include <Wire.h>

// ========== LED 配置 ==========
#define LED_PIN 6
#define LED_COUNT 20
#define MAX_BRIGHTNESS 255

int redValue = 255;
int blueValue = 255;

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// ========== CO₂ 传感器配置 ==========
#define SENSOR_EN_PIN 8
#define READ_INTERVAL_MS 16000
#define SUNRISE_ADDR 0x68
#define WAKEUP_ATTEMPS 1

int16_t co2_value;
uint16_t error_status;
uint8_t meas_mode;
uint16_t meas_period_s;
uint16_t meas_numbers;

// ========== 初始化 LED ==========
void setAllLeds() {
  for (int i = 0; i < LED_COUNT; i++) {
    strip.setPixelColor(i, strip.Color(redValue, 0, blueValue));
  }
  strip.show();
}

// ========== I2C 相关 ==========
void reInitI2C() {
  Wire.begin();
  Wire.setClock(100000);
}

int wakeupSensor(uint8_t target) {
  int attemps = WAKEUP_ATTEMPS;
  int error;
  do {
    Wire.beginTransmission(target);
    error = Wire.endTransmission(true);
  } while (((error != 0) && (error != 2) && (error != 1)) && (--attemps > 0));
  if (error == 4) reInitI2C();
  return attemps;
}

int readSensorConfiguration(uint8_t target) {
  int error = Wire.requestFrom((uint8_t)target, (uint8_t)5, (uint32_t)0x95U, (uint8_t)1, true);
  if (error >= 5) {
    uint8_t byte_hi, byte_lo;
    meas_mode = Wire.read();
    byte_hi = Wire.read(); byte_lo = Wire.read();
    meas_period_s = (((uint16_t)(int8_t)byte_hi) << 8) | (uint16_t)byte_lo;
    byte_hi = Wire.read(); byte_lo = Wire.read();
    meas_numbers = (((int16_t)(int8_t)byte_hi) << 8) | (uint16_t)byte_lo;
  } else {
    error = -1;
  }
  while (Wire.available()) Wire.read();
  return error;
}

void setContinuousMeasurementMode(uint8_t target) {
  if (meas_mode != 0) {
    Wire.beginTransmission(target);
    Wire.write(0x95);
    Wire.write(0x00);
    Wire.endTransmission(true);
  }
}

int readSensorData(uint8_t target) {
  int error = Wire.requestFrom((uint8_t)target, (uint8_t)8, (uint32_t)0x00, (uint8_t)1, true);
  if (error >= 8) {
    uint8_t byte_hi, byte_lo;
    byte_hi = Wire.read(); byte_lo = Wire.read();
    error_status = (((int16_t)(int8_t)byte_hi) << 8) | (uint16_t)byte_lo;
    Wire.read(); Wire.read();  // reserved
    Wire.read(); Wire.read();  // reserved
    byte_hi = Wire.read(); byte_lo = Wire.read();
    co2_value = (((int16_t)(int8_t)byte_hi) << 8) | (uint16_t)byte_lo;
  } else {
    error = -1;
  }
  while (Wire.available()) Wire.read();
  return error;
}

// ========== 串口输入处理 ==========
void handleSerialInput() {
  static String inputLine = "";

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      inputLine.trim();
      if (inputLine.length() > 0) {
        int r = 0, b = 0;
        if (sscanf(inputLine.c_str(), "%d,%d", &r, &b) == 2) {
          redValue = constrain(r, 0, 255);
          blueValue = constrain(b, 0, 255);
          setAllLeds();  // ✅ 设置后不打印任何信息
        }
      }
      inputLine = "";
    } else {
      inputLine += c;
    }
  }
}

// ========== 初始化 ==========
void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  // 初始化 LED
  strip.begin();
  strip.setBrightness(MAX_BRIGHTNESS);
  setAllLeds();

  // 初始化 CO₂ 传感器
  pinMode(SENSOR_EN_PIN, OUTPUT);
  digitalWrite(SENSOR_EN_PIN, HIGH);
  delay(50);
  reInitI2C();
  delay(500);

  if ((wakeupSensor(SUNRISE_ADDR) > 0) && (readSensorConfiguration(SUNRISE_ADDR) > 0)) {
    setContinuousMeasurementMode(SUNRISE_ADDR);
  }
}

// ========== 主循环 ==========
void loop() {
  static unsigned long lastReadTime = 0;

  // 非阻塞定时读取 CO₂
  if (millis() - lastReadTime >= READ_INTERVAL_MS) {
    lastReadTime = millis();

    if (wakeupSensor(SUNRISE_ADDR) > 0) {
      if (readSensorData(SUNRISE_ADDR) > 0) {
        Serial.println(co2_value);  // ✅ 只输出 CO₂ 数字
      }
    }
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));  // 状态灯闪烁
  }

  handleSerialInput();  // 实时接收 r,b 调节红蓝通道
}
