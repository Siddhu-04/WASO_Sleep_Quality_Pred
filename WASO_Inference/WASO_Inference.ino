#include <TensorFlowLite.h>

/*
 * ════════════════════════════════════════════════════════════════════════
 *   WASO Sleep Quality Prediction — On-Chip Inference (TFLite Micro)
 * ════════════════════════════════════════════════════════════════════════
 *
 *   Target: Arduino Nano 33 BLE Sense (nRF52840 + 256 KB RAM)
 *           Also works on: STM32F407, ESP32-S3, Teensy 4.x
 *
 *   What this does:
 *     1. Reads 7 days of HRV + activity + questionnaire features (simulated here)
 *     2. Normalizes them using saved scaler params
 *     3. Quantizes input → runs through int8 CNN → dequantizes output
 *     4. Prints predicted P(awakening tonight) and prediction over Serial
 *
 *   Setup in Arduino IDE:
 *     • Install board: "Arduino Mbed OS Nano Boards" via Boards Manager
 *     • Install library: "Arduino_TensorFlowLite" (Search in Library Manager)
 *     • Place waso_model.h in the same folder as this .ino file
 *     • Select Board: Arduino Nano 33 BLE
 *     • Compile + Upload + Open Serial Monitor @ 9600 baud
 *
 *   Memory footprint:
 *     • Model:   ~10-15 KB Flash
 *     • Tensor arena: ~8 KB RAM
 *     • Total: ~25 KB — leaves 230+ KB free for sensors/BLE
 * ════════════════════════════════════════════════════════════════════════
 */

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "waso_model.h"   // The model header we generated in Colab

// ────────────────────────────────────────────────────────────────────────
// Global TFLite Micro objects
// ────────────────────────────────────────────────────────────────────────
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena: scratch memory used by the interpreter.
// Increase if you see "AllocateTensors() failed" - try 16, 24, 32 KB.
constexpr int kTensorArenaSize = 12 * 1024;   // 12 KB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ────────────────────────────────────────────────────────────────────────
// SETUP
// ────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(9600);
  while (!Serial && millis() < 5000);   // Wait for Serial (max 5 s)

  Serial.println(F("╔════════════════════════════════════════════╗"));
  Serial.println(F("║   WASO Sleep Quality Prediction — On-Chip  ║"));
  Serial.println(F("╚════════════════════════════════════════════╝"));

  // 1. Set up error reporter
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // 2. Load the model
  model = tflite::GetModel(waso_model_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print(F("ERROR: Model schema mismatch. Expected "));
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(F(" got "));
    Serial.println(model->version());
    while (1);
  }
  Serial.println(F("✓ Model loaded"));

  // 3. Register operations (AllOpsResolver loads all ops — easiest for prototyping)
  // For production, use MicroMutableOpResolver to save flash space (~50 KB savings).
  static tflite::AllOpsResolver resolver;

  // 4. Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 5. Allocate tensors from the arena
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println(F("ERROR: AllocateTensors() failed. Try increasing kTensorArenaSize."));
    while (1);
  }
  Serial.println(F("✓ Tensors allocated"));

  // 6. Get input/output handles
  input  = interpreter->input(0);
  output = interpreter->output(0);

  // Sanity checks
  Serial.print(F("Input shape:  "));
  for (int i = 0; i < input->dims->size; ++i) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(F(" × "));
  }
  Serial.println();

  Serial.print(F("Input type:   "));
  Serial.println(input->type == kTfLiteInt8 ? F("int8") : F("other"));
  Serial.print(F("Arena used:   "));
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(F(" bytes"));
  Serial.println();
}

// ────────────────────────────────────────────────────────────────────────
// HELPER — z-score normalize, then quantize to int8
// ────────────────────────────────────────────────────────────────────────
int8_t quantize_input(float raw_value, int feature_idx) {
  // Step 1: z-score normalize using saved scaler params
  float normalized = (raw_value - scaler_mean[feature_idx]) / scaler_std[feature_idx];
  // Step 2: quantize to int8
  int q = (int)round(normalized / input_scale) + input_zero_point;
  if (q < -128) q = -128;
  if (q >  127) q =  127;
  return (int8_t)q;
}

float dequantize_output(int8_t raw) {
  return ((float)raw - output_zero_point) * output_scale;
}

// ────────────────────────────────────────────────────────────────────────
// PREDICTION FUNCTION
// Input: 7 days × 7 features (raw, unscaled values from sensors/questionnaires)
// Output: probability of awakening (0.0 to 1.0)
// ────────────────────────────────────────────────────────────────────────
float predict_waso(float window[WINDOW_SIZE][NUM_FEATURES]) {
  // 1. Fill input tensor (normalize + quantize each value)
  for (int d = 0; d < WINDOW_SIZE; ++d) {
    for (int f = 0; f < NUM_FEATURES; ++f) {
      int idx = d * NUM_FEATURES + f;
      input->data.int8[idx] = quantize_input(window[d][f], f);
    }
  }

  // 2. Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println(F("ERROR: Invoke() failed"));
    return -1.0f;
  }

  // 3. Dequantize output to get probability
  int8_t raw_out = output->data.int8[0];
  return dequantize_output(raw_out);
}

// ────────────────────────────────────────────────────────────────────────
// LOOP — run a demo prediction every 5 seconds
// ────────────────────────────────────────────────────────────────────────
void loop() {
  // ── DEMO DATA ──────────────────────────────────────────────────────
  // In production: read these from real sensors (HRV from PPG, steps from
  // accelerometer, questionnaire scores from user input).
  // For demo: simulate a "stressed person" with high LF/HF, low RMSSD.
  //
  // Feature order: lf_hf, rmssd, steps, accel, gyro, isi, whoqol
  //
  float demo_window[WINDOW_SIZE][NUM_FEATURES] = {
    // Day 1
    {0.85f, 90.0f,  3200.0f, 1.5f, 2.0f, 14.0f, 82.0f},
    // Day 2
    {0.78f, 95.0f,  3800.0f, 1.8f, 2.3f, 14.0f, 82.0f},
    // Day 3
    {0.92f, 80.0f,  2900.0f, 1.2f, 1.8f, 14.0f, 82.0f},
    // Day 4
    {0.88f, 85.0f,  3100.0f, 1.4f, 2.0f, 14.0f, 82.0f},
    // Day 5
    {0.95f, 75.0f,  2800.0f, 1.1f, 1.7f, 14.0f, 82.0f},
    // Day 6 (weekend - rest day)
    {0.65f, 130.0f, 5200.0f, 2.5f, 3.0f, 14.0f, 82.0f},
    // Day 7
    {0.80f, 100.0f, 4000.0f, 2.0f, 2.5f, 14.0f, 82.0f}
  };

  // ── RUN INFERENCE ──────────────────────────────────────────────────
  unsigned long t_start = micros();
  float p_awakening = predict_waso(demo_window);
  unsigned long t_elapsed = micros() - t_start;

  // ── REPORT ─────────────────────────────────────────────────────────
  Serial.println(F("─────────────────────────────────────────"));
  Serial.print(F("Inference time: "));
  Serial.print(t_elapsed);
  Serial.println(F(" µs"));

  Serial.print(F("P(awakening tonight): "));
  Serial.println(p_awakening, 4);

  Serial.print(F("Prediction: "));
  if (p_awakening > 0.5f) {
    Serial.println(F("⚠ Likely DISRUPTED sleep tonight"));
  } else {
    Serial.println(F("✓ Likely UNINTERRUPTED sleep"));
  }
  Serial.println();

  delay(5000);
}
