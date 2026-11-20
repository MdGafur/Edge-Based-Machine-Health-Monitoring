#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ESP_I2S.h>
#include <math.h>

// ---------------- PIN CONFIGURATION ----------------

#define SDA_PIN 21
#define SCL_PIN 22
#define TEMP_PIN 4

#define I2S_SCK 26
#define I2S_WS  25
#define I2S_SD  33

// ---------------- SAMPLING CONFIGURATION ----------------

#define ACCEL_HZ 100
#define ACCEL_INTERVAL_US (1000000UL / ACCEL_HZ)
#define ACCEL_WINDOW 100

#define MIC_SAMPLE_RATE 16000
// 0 = left slot (INMP441 L/R pin on GND), 1 = right slot (L/R pin on 3.3V)
#define MIC_CHANNEL_OFFSET 0
// Bytes the microphone produces in one accelerometer interval (stereo, 32-bit).
#define MIC_BYTES_PER_SLOT (MIC_SAMPLE_RATE * 2 * 4 / ACCEL_HZ)
#define MIC_SAMPLES_PER_SLOT (MIC_BYTES_PER_SLOT / 4)

// ---------------- SENSOR OBJECTS ----------------

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
Adafruit_INA219 ina219;

OneWire oneWire(TEMP_PIN);
DallasTemperature tempSensor(&oneWire);

I2SClass I2S;

// ---------------- SENSOR STATUS ----------------

bool adxlOK = false;
bool inaOK = false;
bool tempOK = false;
bool micOK = false;

// ---------------- DATA BUFFERS ----------------

float axData[ACCEL_WINDOW];
float ayData[ACCEL_WINDOW];
float azData[ACCEL_WINDOW];

float temperatureC = NAN;
unsigned long lastTempRequest = 0;

const unsigned long TEMP_CONVERSION_MS = 750;

// ---------------- MICROPHONE ACCUMULATION ----------------

int32_t micBuffer[MIC_SAMPLES_PER_SLOT];

double micSum = 0.0;
double micSumSquares = 0.0;
unsigned long micCount = 0;
bool micStalled = false;

void resetMicrophoneWindow() {

  micSum = 0.0;
  micSumSquares = 0.0;
  micCount = 0;
  micStalled = false;
}

void accumulateMicrophoneSlot() {

  if (!micOK || micStalled) return;

  size_t bytesRead =
      I2S.readBytes((char *)micBuffer, MIC_BYTES_PER_SLOT);

  size_t count = bytesRead / sizeof(int32_t);

  if (count == 0) {
    micStalled = true;
    return;
  }

  for (size_t i = MIC_CHANNEL_OFFSET; i < count; i += 2) {

    double sample = (double)(micBuffer[i] >> 8);

    micSum += sample;
    micSumSquares += sample * sample;
    micCount++;
  }
}

float finishMicrophoneRMS() {

  if (micCount == 0) return NAN;

  double mean = micSum / micCount;
  double variance = micSumSquares / micCount - mean * mean;

  if (variance < 0.0) variance = 0.0;

  return (float)sqrt(variance);
}

// ---------------- ACCELEROMETER FUNCTION ----------------

bool collectAccelerationWindow(
    float &meanX,
    float &meanY,
    float &meanZ,
    float &vibrationRMS,
    float &vibrationPeakToPeak,
    float &vibrationXRMS,
    float &vibrationYRMS,
    float &vibrationZRMS,
    unsigned long &windowDuration
) {

  if (!adxlOK) return false;

  resetMicrophoneWindow();

  unsigned long startTime = micros();
  unsigned long nextSample = startTime;

  for (int i = 0; i < ACCEL_WINDOW; i++) {

    while ((int32_t)(micros() - nextSample) < 0) {
      // Wait for next sampling instant.
    }

    sensors_event_t event;

    accel.getEvent(&event);

    axData[i] = event.acceleration.x;
    ayData[i] = event.acceleration.y;
    azData[i] = event.acceleration.z;

    accumulateMicrophoneSlot();

    nextSample += ACCEL_INTERVAL_US;
  }

  windowDuration = micros() - startTime;

  double sumX = 0;
  double sumY = 0;
  double sumZ = 0;

  for (int i = 0; i < ACCEL_WINDOW; i++) {

    sumX += axData[i];
    sumY += ayData[i];
    sumZ += azData[i];
  }

  meanX = sumX / ACCEL_WINDOW;
  meanY = sumY / ACCEL_WINDOW;
  meanZ = sumZ / ACCEL_WINDOW;

  double sumVibration = 0;
  double sumVX = 0;
  double sumVY = 0;
  double sumVZ = 0;

  float minimumMagnitude = 1000000.0;
  float maximumMagnitude = 0.0;

  for (int i = 0; i < ACCEL_WINDOW; i++) {

    float dx = axData[i] - meanX;
    float dy = ayData[i] - meanY;
    float dz = azData[i] - meanZ;

    sumVX += dx * dx;
    sumVY += dy * dy;
    sumVZ += dz * dz;

    float magnitude =
        sqrt(dx * dx + dy * dy + dz * dz);

    sumVibration += magnitude * magnitude;

    if (magnitude < minimumMagnitude) {
      minimumMagnitude = magnitude;
    }

    if (magnitude > maximumMagnitude) {
      maximumMagnitude = magnitude;
    }
  }

  vibrationXRMS = sqrt(sumVX / ACCEL_WINDOW);
  vibrationYRMS = sqrt(sumVY / ACCEL_WINDOW);
  vibrationZRMS = sqrt(sumVZ / ACCEL_WINDOW);

  vibrationRMS = sqrt(sumVibration / ACCEL_WINDOW);

  vibrationPeakToPeak =
      maximumMagnitude - minimumMagnitude;

  return true;
}

// ---------------- SETUP ----------------

void setup() {

  Serial.begin(115200);
  delay(1500);

  Wire.begin(SDA_PIN, SCL_PIN);

  adxlOK = accel.begin();

  if (adxlOK) {

    accel.setRange(ADXL345_RANGE_16_G);

    accel.setDataRate(ADXL345_DATARATE_100_HZ);
  }

  inaOK = ina219.begin();

  tempSensor.begin();

  tempOK = tempSensor.getDeviceCount() > 0;

  if (tempOK) {

    tempSensor.setWaitForConversion(false);

    tempSensor.requestTemperatures();

    lastTempRequest = millis();
  }

  I2S.setPins(I2S_SCK, I2S_WS, -1, I2S_SD);

  micOK = I2S.begin(
      I2S_MODE_STD,
      MIC_SAMPLE_RATE,
      I2S_DATA_BIT_WIDTH_32BIT,
      I2S_SLOT_MODE_STEREO
  );

  Serial.println(
    "timestamp_ms,"
    "acceleration_x_m_s2,"
    "acceleration_y_m_s2,"
    "acceleration_z_m_s2,"
    "bus_voltage_v,"
    "current_mA,"
    "temperature_C,"
    "microphone_rms_raw,"
    "vibration_rms,"
    "vibration_peak_to_peak,"
    "vibration_x_rms,"
    "vibration_y_rms,"
    "vibration_z_rms,"
    "accel_samples,"
    "accel_window_ms"
  );
}

// ---------------- MAIN LOOP ----------------

void loop() {

  float ax = NAN;
  float ay = NAN;
  float az = NAN;

  float vibrationRMS = NAN;
  float vibrationPeakToPeak = NAN;

  float vibrationXRMS = NAN;
  float vibrationYRMS = NAN;
  float vibrationZRMS = NAN;

  float busVoltage = NAN;
  float currentmA = NAN;
  float microphoneRMS = NAN;

  unsigned long accelWindowDuration = 0;

  // 1. Collect accelerometer window (microphone is sampled in the same window)

  bool accelSuccess = collectAccelerationWindow(
      ax,
      ay,
      az,
      vibrationRMS,
      vibrationPeakToPeak,
      vibrationXRMS,
      vibrationYRMS,
      vibrationZRMS,
      accelWindowDuration
  );

  microphoneRMS = finishMicrophoneRMS();

  // 2. Read INA219

  if (inaOK) {

    busVoltage = ina219.getBusVoltage_V();

    currentmA = ina219.getCurrent_mA();
  }

  // 3. Update DS18B20 asynchronously

  unsigned long now = millis();

  if (tempOK) {

    if (now - lastTempRequest >= TEMP_CONVERSION_MS) {

      float t = tempSensor.getTempCByIndex(0);

      if (t != DEVICE_DISCONNECTED_C &&
          t > -100.0f && t < 150.0f) {

        temperatureC = t;
      }

      tempSensor.requestTemperatures();

      lastTempRequest = now;
    }
  }

  // 4. Print one CSV row

  Serial.printf(
    "%lu,"
    "%.3f,%.3f,%.3f,"
    "%.3f,%.3f,%.2f,%.2f,"
    "%.5f,%.5f,"
    "%.5f,%.5f,%.5f,"
    "%d,%.2f\n",

    (unsigned long)millis(),

    ax, ay, az,

    busVoltage,
    currentmA,
    temperatureC,
    microphoneRMS,

    vibrationRMS,
    vibrationPeakToPeak,

    vibrationXRMS,
    vibrationYRMS,
    vibrationZRMS,

    accelSuccess ? ACCEL_WINDOW : 0,

    accelWindowDuration / 1000.0f
  );
}
