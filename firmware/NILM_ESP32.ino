/**
 * NILM_ESP32.ino — NILM Classifier + Real Electrical Measurement
 * ─────────────────────────────────────────────────────────────────
 * Hardware:
 *   GPIO 34 — SCT-013-000 (current): NILM + electrical measurement
 *   GPIO 35 — ZMPT101B   (voltage): electrical measurement
 *   2 × 10 kΩ bias  |  22 µF decoupling  |  1 µF ADC filter
 *
 * Dual pipeline:
 *   1. NILM (8192 Hz timer) → GPIO34 circular buffer → MLP →
 *      temporal filter → labels [ventilador, liquidificador, secador]
 *   2. Electrical measurement (every SEND_INTERVAL_MS) →
 *      1000 V+I samples @ 10 kHz → Vrms, Irms, P, S, PF
 *   3. MongoDB POST: NILM + electrical measurement + energy + cost
 *
 * Calibration (determined using a multimeter):
 *   CURRENT_CALIBRATION = 0.00438  (LSB → A)
 *   VOLTAGE_CALIBRATION = 0.58424  (LSB → V)
 */

#include "nilm_model.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>

// ═════════════════════════════════════════════════════════════════
//  CONFIGURATION — fill in before compiling
// ═════════════════════════════════════════════════════════════════
#define WIFI_SSID         "YOUR_WIFI_SSID"
#define WIFI_PASSWORD     "YOUR_WIFI_PASSWORD"
#define SERVER_URL        "http://YOUR_SERVER_IP:3000/measure"

// ── Pins ─────────────────────────────────────────────────────────
#define CURRENT_PIN       34    // SCT-013-000
#define VOLTAGE_PIN       35    // ZMPT101B

// ── Timer NILM ────────────────────────────────────────────────────
#define SERIAL_BAUD       115200
#define TIMER_BASE_FREQ   1000000        // 1 MHz
#define TIMER_ALARM_CNT   122            // 1 MHz / 122 ≈ 8196 Hz

// ── Intervals ────────────────────────────────────────────────────
#define SEND_INTERVAL_MS  1000
#define PRINT_INTERVAL_MS 1000

// ── NTP / Tariff ─────────────────────────────────────────────────
#define NTP_SERVER        "pool.ntp.org"
#define GMT_OFFSET_S      (-3 * 3600)    // UTC-3 Brasília
#define DST_OFFSET_S      0
#define TARIFA_KWH        0.95f          // BRL/kWh

// ── Sensor calibration ───────────────────────────────────────
#define CURRENT_CALIBRATION  0.00438f    // LSB → A  (10 kHz, 1000 samples)
#define VOLTAGE_CALIBRATION  0.72418f    // LSB → V  (calibrated: 141.55V→127V)

// ── Electrical measurement ─────────────────────────────────────────────
#define ELEC_SAMPLES      1000           // samples per measurement cycle
#define ELEC_INTERVAL_US  100            // µs — 10 kHz target
#define CURRENT_NOISE_THR 0.08f          // A — values below this threshold are considered 0
#define VOLTAGE_NOISE_THR 10.0f          // V — values below this threshold are considered 0
#define MEAS_ALPHA        0.2f           // exponential moving average coefficient (smooths Vrms/Irms)

// ── NILM temporal filter ─────────────────────────────────────────
#define SMOOTH_K          10             // windows for moving average (~156 ms)
#define THRESH_ON         0.65f          // minimum probability to declare ON
#define THRESH_OFF        0.35f          // maximum probability to declare OFF
#define DEBOUNCE_WIN      3              // confirmations required to change state

// ── Real ACTIVE power per device (calibration + measured data)
static const char*  DEV_NAME[]   = {"ventilador","liquidificador","secador"};
static const float  POWER_EST[]  = {60.0f, 186.0f, 180.0f};   // W

// ═════════════════════════════════════════════════════════════════
//  GLOBAL VARIABLES
// ═════════════════════════════════════════════════════════════════

// ── NILM circular buffer (GPIO 34) ───────────────────────────────
#define CIRC_SIZE  512
#define CIRC_MASK  (CIRC_SIZE-1)
static float             circ_buf[CIRC_SIZE];
static volatile uint32_t circ_write = 0;

// ── Timer ────────────────────────────────────────────────────────
static hw_timer_t     *g_timer = NULL;
static portMUX_TYPE    g_mux   = portMUX_INITIALIZER_UNLOCKED;
static volatile bool   g_tick  = false;
static volatile uint8_t g_step = 0;

// ── NILM — inference and temporal filter ──────────────────────────
static float   g_window[NILM_WINDOW_SIZE];
static float   g_probs_raw[NILM_N_OUTPUTS];
static float   g_prob_hist[NILM_N_OUTPUTS][SMOOTH_K];
static uint8_t g_hist_idx  = 0;
static uint8_t g_hist_cnt  = 0;
static float   g_probs_avg[NILM_N_OUTPUTS];
static uint8_t g_state        [NILM_N_OUTPUTS] = {0};
static uint8_t g_pending      [NILM_N_OUTPUTS] = {0};
static uint8_t g_debounce_cnt [NILM_N_OUTPUTS] = {0};

// ── Real electrical measurement ────────────────────────────────────────
struct ElecMeas {
    float voltageRms;      // V (smoothed)
    float currentRms;      // A (smoothed)
    float powerW;          // W (real active power)
    float apparentVA;      // VA (apparent power)
    float powerFactor;     // PF = P / S
    float voltageAdcRms;   // LSB (for debugging)
    float currentAdcRms;   // LSB (for debugging)
};
static ElecMeas g_meas     = {127.0f, 0, 0, 0, 0, 0, 0};
static float    g_avg_vrms = 127.0f;   // initialized with nominal value
static float    g_avg_irms = 0.0f;
static bool     g_meas_first = true;

// ── Energy and cost ──────────────────────────────────────────────
static float         g_energy[NILM_N_OUTPUTS] = {0};
static float         g_energy_total = 0;
static float         g_cost_total   = 0;
static unsigned long g_last_energy_ms = 0;

// ── Timing control ────────────────────────────────────────────
static uint32_t g_last_send_ms  = 0;
static uint32_t g_last_print_ms = 0;
static bool     g_ntp_ok        = false;

// ═════════════════════════════════════════════════════════════════
//  ISR — Timer NILM
// ═════════════════════════════════════════════════════════════════
void IRAM_ATTR onTimerISR() {
    portENTER_CRITICAL_ISR(&g_mux);
    g_tick = true;
    portEXIT_CRITICAL_ISR(&g_mux);
}

// ═════════════════════════════════════════════════════════════════
//  REAL ELECTRICAL MEASUREMENT (V + I — 10 kHz, 1000 samples)
// ═════════════════════════════════════════════════════════════════
static void measure_electrical() {

    // ── Step 1: calculate DC offsets ──────────────────────────────
    double sumV = 0, sumI = 0;
    for (int i = 0; i < ELEC_SAMPLES; i++) {
        sumV += analogRead(VOLTAGE_PIN);
        sumI += analogRead(CURRENT_PIN);
        delayMicroseconds(ELEC_INTERVAL_US);
    }
    float offV = (float)(sumV / ELEC_SAMPLES);
    float offI = (float)(sumI / ELEC_SAMPLES);

    // ── Step 2: calculate RMS and real power ─────────────────────
    double sumV2 = 0, sumI2 = 0, sumP = 0;
    for (int i = 0; i < ELEC_SAMPLES; i++) {
        float v  = (float)analogRead(VOLTAGE_PIN) - offV;
        float ci = (float)analogRead(CURRENT_PIN) - offI;
        sumV2 += (double)(v  * v);
        sumI2 += (double)(ci * ci);
        sumP  += (double)(v  * ci);     // instantaneous product v(t)×i(t)
        delayMicroseconds(ELEC_INTERVAL_US);
    }

    float adcVrms = sqrtf((float)(sumV2 / ELEC_SAMPLES));
    float adcIrms = sqrtf((float)(sumI2 / ELEC_SAMPLES));

    // ── Step 3: convert to physical units ──────────────────
    float vRaw = adcVrms * VOLTAGE_CALIBRATION;
    float iRaw = adcIrms * CURRENT_CALIBRATION;

    // Noise threshold
    if (iRaw < CURRENT_NOISE_THR) iRaw = 0.0f;
    if (vRaw < VOLTAGE_NOISE_THR) vRaw = 127.0f;  // fallback: nominal mains voltage

    // ── Step 4: exponential moving average (stabilizes display) ───
    if (g_meas_first) {
        g_avg_vrms = vRaw;
        g_avg_irms = iRaw;
        g_meas_first = false;
    } else {
        g_avg_vrms = MEAS_ALPHA * vRaw + (1.0f - MEAS_ALPHA) * g_avg_vrms;
        g_avg_irms = MEAS_ALPHA * iRaw + (1.0f - MEAS_ALPHA) * g_avg_irms;
    }

    // ── Step 5: real power = mean of v(t)×i(t) × calibration factors
    float pRaw = fabsf((float)(sumP / ELEC_SAMPLES)
                       * VOLTAGE_CALIBRATION * CURRENT_CALIBRATION);
    if (g_avg_irms == 0.0f) pRaw = 0.0f;

    // ── Step 6: apparent power and power factor ───────────
    float S  = g_avg_vrms * g_avg_irms;
    float fp = (S > 0.5f) ? constrain(pRaw / S, 0.0f, 1.0f) : 0.0f;

    // ── Store result ─────────────────────────────────────────
    g_meas.voltageRms    = g_avg_vrms;
    g_meas.currentRms    = g_avg_irms;
    g_meas.powerW        = pRaw;
    g_meas.apparentVA    = S;
    g_meas.powerFactor   = fp;
    g_meas.voltageAdcRms = adcVrms;
    g_meas.currentAdcRms = adcIrms;
}

// ═════════════════════════════════════════════════════════════════
//  NILM TEMPORAL FILTER (moving average + hysteresis + debounce)
// ═════════════════════════════════════════════════════════════════
static void apply_temporal_filter() {
    // 1. Store probabilities
    for (int i = 0; i < NILM_N_OUTPUTS; i++)
        g_prob_hist[i][g_hist_idx] = g_probs_raw[i];
    g_hist_idx = (g_hist_idx + 1) % SMOOTH_K;
    if (g_hist_cnt < SMOOTH_K) g_hist_cnt++;

    // 2. Moving average
    for (int i = 0; i < NILM_N_OUTPUTS; i++) {
        float sum = 0;
        for (int k = 0; k < g_hist_cnt; k++) sum += g_prob_hist[i][k];
        g_probs_avg[i] = sum / g_hist_cnt;
    }

    // 3. Hysteresis + debounce
    for (int i = 0; i < NILM_N_OUTPUTS; i++) {
        uint8_t novo;
        if      (g_probs_avg[i] >= THRESH_ON)  novo = 1;
        else if (g_probs_avg[i] <= THRESH_OFF)  novo = 0;
        else                                     novo = g_state[i];

        if (novo == g_pending[i]) {
            if (g_debounce_cnt[i] < DEBOUNCE_WIN) g_debounce_cnt[i]++;
            if (g_debounce_cnt[i] >= DEBOUNCE_WIN && novo != g_state[i])
                g_state[i] = novo;
        } else {
            g_pending[i]      = novo;
            g_debounce_cnt[i] = 1;
        }
    }
}

// ═════════════════════════════════════════════════════════════════
//  ENERGY — distributes measured real power among active devices
// ═════════════════════════════════════════════════════════════════
static void update_energy() {
    unsigned long now = millis();
    float h = (now - g_last_energy_ms) / 3600000.0f;
    g_last_energy_ms = now;

    // Sum of estimated power of active devices
    float sum_est = 0;
    for (int i = 0; i < NILM_N_OUTPUTS; i++)
        sum_est += g_state[i] ? POWER_EST[i] : 0.0f;

    for (int i = 0; i < NILM_N_OUTPUTS; i++) {
        if (!g_state[i]) continue;
        // If a real measurement is available, distribute it proportionally
        // Otherwise, use the table estimate
        float ratio  = (sum_est > 0) ? POWER_EST[i] / sum_est : 1.0f;
        float p_dev  = (g_meas.powerW > 1.0f)
                       ? g_meas.powerW * ratio
                       : POWER_EST[i];
        g_energy[i]    += p_dev * h / 1000.0f;
        g_energy_total += p_dev * h / 1000.0f;
    }
    g_cost_total = g_energy_total * TARIFA_KWH;
}

// ═════════════════════════════════════════════════════════════════
//  TIMESTAMP
// ═════════════════════════════════════════════════════════════════
static void get_timestamp(char *buf, size_t n) {
    if (g_ntp_ok) {
        struct tm ti;
        if (getLocalTime(&ti)) { strftime(buf, n, "%Y-%m-%d %H:%M:%S", &ti); return; }
    }
    snprintf(buf, n, "ms:%lu", millis());
}

// ═════════════════════════════════════════════════════════════════
//  POST MONGODB
// ═════════════════════════════════════════════════════════════════
static void send_to_server() {
    if (WiFi.status() != WL_CONNECTED) return;

    char ts[32]; get_timestamp(ts, sizeof(ts));

    String json = "{";
    json += "\"timestamp\":\"" + String(ts) + "\",";
    json += "\"ms\":"          + String(millis()) + ",";

    // ── Device array (NILM) ──────────────────────────────
    json += "\"devices\":[";
    for (int i = 0; i < NILM_N_OUTPUTS; i++) {
        float sum_est = 0;
        for (int j = 0; j < NILM_N_OUTPUTS; j++)
            sum_est += g_state[j] ? POWER_EST[j] : 0.0f;

        // Device power: measured real power (distributed) or estimated
        float ratio  = (sum_est > 0 && g_state[i]) ? POWER_EST[i] / sum_est : 0.0f;
        float p_dev  = g_state[i]
                       ? ((g_meas.powerW > 1.0f) ? g_meas.powerW * ratio : POWER_EST[i])
                       : 0.0f;
        float i_dev  = g_state[i]
                       ? ((g_meas.currentRms > CURRENT_NOISE_THR)
                          ? g_meas.currentRms * ratio
                          : POWER_EST[i] / g_meas.voltageRms)
                       : 0.0f;
        float va_dev = i_dev * g_meas.voltageRms;
        float fp_dev = (va_dev > 0) ? constrain(p_dev / va_dev, 0.0f, 1.0f) : 0.0f;

        json += "{";
        json += "\"name\":\""       + String(DEV_NAME[i])      + "\",";
        json += "\"active\":"       + String(g_state[i])        + ",";
        json += "\"prob\":"         + String(g_probs_avg[i], 3) + ",";
        json += "\"powerW\":"       + String(p_dev, 1)          + ",";
        json += "\"currentRms\":"   + String(i_dev, 4)          + ",";
        json += "\"voltageRms\":"   + String(g_meas.voltageRms, 2) + ",";
        json += "\"powerFactor\":"  + String(fp_dev, 3)         + ",";
        json += "\"energyKwh\":"    + String(g_energy[i], 6)    + ",";
        json += "\"costR$\":"       + String(g_energy[i] * TARIFA_KWH, 6);
        json += (i < NILM_N_OUTPUTS-1) ? "}," : "}";
    }
    json += "],";

    // ── Total electrical measurement (physical sensors) ───────────────────
    json += "\"electrical\":{";
    json += "\"voltageRms\":"    + String(g_meas.voltageRms,  2) + ",";
    json += "\"currentRms\":"    + String(g_meas.currentRms,  4) + ",";
    json += "\"powerW\":"        + String(g_meas.powerW,      2) + ",";
    json += "\"apparentVA\":"    + String(g_meas.apparentVA,  2) + ",";
    json += "\"powerFactor\":"   + String(g_meas.powerFactor, 3) + ",";
    json += "\"energyKwh\":"     + String(g_energy_total,     6) + ",";
    json += "\"costPerKwh\":"    + String(TARIFA_KWH,         2) + ",";
    json += "\"totalCost\":"     + String(g_cost_total,       6) + ",";
    json += "\"adcVoltageRms\":" + String(g_meas.voltageAdcRms, 2) + ",";
    json += "\"adcCurrentRms\":" + String(g_meas.currentAdcRms, 2);
    json += "},";

    json += "\"source\":\"esp32_nilm_s3\"}";

    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);
    int code = http.POST(json);
    http.end();
    if (code != 201) Serial.printf("[HTTP %d]\n", code);
}

// ═════════════════════════════════════════════════════════════════
//  SERIAL
// ═════════════════════════════════════════════════════════════════
static void print_serial() {
    char ts[32]; get_timestamp(ts, sizeof(ts));
    Serial.printf("[%s] ", ts);

    bool any = false;
    for (int i = 0; i < NILM_N_OUTPUTS; i++) {
        if (g_state[i]) {
            if (any) Serial.print(" + ");
            Serial.printf("%s (p=%.2f)", DEV_NAME[i], g_probs_avg[i]);
            any = true;
        }
    }
    if (!any) Serial.print("None");

    Serial.printf(
        " | V=%.1fV  I=%.3fA  P=%.1fW  S=%.1fVA  fp=%.2f"
        " | E=%.5fkWh  R$%.4f\n",
        g_meas.voltageRms, g_meas.currentRms,
        g_meas.powerW,     g_meas.apparentVA, g_meas.powerFactor,
        g_energy_total,    g_cost_total
    );
}

// ═════════════════════════════════════════════════════════════════
//  SETUP
// ═════════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(SERIAL_BAUD); delay(400);

    Serial.println("╔════════════════════════════════════════════════╗");
    Serial.println("║  NILM ESP32     |  Real V+I measurement           ║");
    Serial.println("║  GPIO 34: SCT-013  |  GPIO 35: ZMPT101B       ║");
    Serial.printf ("║  Cal: I=%.5f A/LSB  V=%.5f V/LSB   ║\n",
                   CURRENT_CALIBRATION, VOLTAGE_CALIBRATION);
    Serial.println("╠════════════════════════════════════════════════╣");

    // ── ADC ──────────────────────────────────────────────────────
    analogReadResolution(12);
    analogSetPinAttenuation(CURRENT_PIN, ADC_11db);
    analogSetPinAttenuation(VOLTAGE_PIN, ADC_11db);
    for (int i = 0; i < 64; i++) { analogRead(CURRENT_PIN); analogRead(VOLTAGE_PIN); }

    float bias = 0;
    for (int i = 0; i < 256; i++) bias += analogRead(CURRENT_PIN);
    bias /= 256.0f;
    for (int i = 0; i < CIRC_SIZE; i++) circ_buf[i] = bias;
    Serial.printf("║  ADC bias (GPIO34): %.0f LSB                    ║\n", bias);

    // ── NILM history ───────────────────────────────────────────
    memset(g_prob_hist, 0, sizeof(g_prob_hist));

    // ── Wi-Fi ────────────────────────────────────────────────────
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis()-t0 < 12000) {
        delay(300); Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("║  WiFi OK  %s\n", WiFi.localIP().toString().c_str());
        configTime(GMT_OFFSET_S, DST_OFFSET_S, NTP_SERVER);
        struct tm ti; g_ntp_ok = getLocalTime(&ti, 5000);
        if (g_ntp_ok) {
            char b[32]; strftime(b, 32, "%Y-%m-%d %H:%M:%S", &ti);
            Serial.printf("║  NTP OK  %s\n", b);
        }
    } else {
        Serial.println("║  WiFi FAILED — offline mode");
        WiFi.mode(WIFI_OFF);
    }

    // ── Timer NILM ───────────────────────────────────────────────
    g_timer = timerBegin(TIMER_BASE_FREQ);
    timerAttachInterrupt(g_timer, &onTimerISR);
    timerAlarm(g_timer, TIMER_ALARM_CNT, true, 0);
    Serial.printf("║  Timer NILM: fs=%.0f Hz\n",
                  (float)TIMER_BASE_FREQ / TIMER_ALARM_CNT);

    // ── Initial measurement ──────────────────────────────────────────
    Serial.println("║  Performing initial V+I measurement...");
    measure_electrical();
    Serial.printf("║  V=%.1fV  I=%.3fA  P=%.1fW  fp=%.2f\n",
                  g_meas.voltageRms, g_meas.currentRms,
                  g_meas.powerW,     g_meas.powerFactor);

    Serial.println("╚════════════════════════════════════════════════╝\n");

    g_last_energy_ms = g_last_send_ms = g_last_print_ms = millis();
}

// ═════════════════════════════════════════════════════════════════
//  LOOP
// ═════════════════════════════════════════════════════════════════
void loop() {

    // ── Step 1: sample ADC (NILM, via timer tick) ──────────
    bool tick;
    portENTER_CRITICAL(&g_mux);
    tick = g_tick; g_tick = false;
    portEXIT_CRITICAL(&g_mux);

    if (tick) {
        circ_buf[circ_write & CIRC_MASK] = (float)analogRead(CURRENT_PIN);
        circ_write++;

        // ── Step 2: classify after accumulating STEP samples ────
        if (++g_step >= NILM_STEP_SIZE) {
            g_step = 0;
            if (circ_write >= (uint32_t)NILM_WINDOW_SIZE) {
                uint32_t base = circ_write - NILM_WINDOW_SIZE;
                for (int i = 0; i < NILM_WINDOW_SIZE; i++)
                    g_window[i] = circ_buf[(base + i) & CIRC_MASK];

                uint8_t lbl_raw[NILM_N_OUTPUTS];
                nilm_run_s3(g_window, g_probs_raw, lbl_raw);
                apply_temporal_filter();
            }
        }
    }

    uint32_t now = millis();

    // ── Step 3: measure V+I and send to server ──────────────────
    if (now - g_last_send_ms >= SEND_INTERVAL_MS) {
        g_last_send_ms = now;

        // Real electrical measurement (~200 ms blocking time)
        measure_electrical();

        // Accumulate energy using real power
        update_energy();

        // Send to MongoDB
        send_to_server();
    }

    // ── Step 4: print to Serial ─────────────────────────────────
    if (now - g_last_print_ms >= PRINT_INTERVAL_MS) {
        g_last_print_ms = now;
        print_serial();
    }
}

/*
 * NOTES:
 *
 * ADC GPIO 34 (current):
 *   • Used by the NILM timer at 8196 Hz through the circular buffer
 *   • Also sampled directly during electrical measurement (1000 × 10 kHz)
 *   • Both uses are sequential (not simultaneous) — no conflict
 *
 * ADC GPIO 35 (voltage):
 *   • Used only for electrical measurement (1000 samples @ 10 kHz)
 *   • ZMPT101B with internal 2.5 V offset → bias ≈ 3000 LSB (normal)
 *
 * Electrical measurement (~200 ms per cycle):
 *   • During measurement, the loop is blocked
 *   • The ISR timer keeps firing, but g_tick stores only one state
 *   • NILM resumes normally after the measurement
 *   • Impact: up to 1–2 NILM windows lost per second (acceptable)
 *
 * Distributed energy:
 *   • Uses measured REAL power (P = mean(v×i) × calibration factors)
 *   • Distributes it proportionally according to the estimated power of active devices
 *   • Example: fan (60 W) + hair dryer (180 W) active, P_real=230 W
 *         → fan receives 230 × 60/240 = 57.5 W
 *         → hair dryer receives 230 × 180/240 = 172.5 W
 *
 * Timer API v3.x (ESP32 Arduino Core ≥ 3.0):
 *   timerBegin(freq)  |  timerAttachInterrupt(t, isr)  |  timerAlarm(t, cnt, true, 0)
 */
