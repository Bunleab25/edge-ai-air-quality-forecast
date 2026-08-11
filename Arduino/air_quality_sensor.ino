// =========================================================================
// AeroSense Edge Node: ESP32-S3 + PMS7003 + BME280 + @RMUTI-One + MQTT
// Anthor Bunleab Chea
// 02/08/22006
// =========================================================================

#include <WiFi.h>
#include "esp_eap_client.h" 
#include "esp_wifi.h"       
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// --- Wi-Fi Credentials (WPA2-Enterprise) ---
const char* WIFI_SSID     = "@RMUTI-One";
const char* WIFI_IDENTITY = "bunleab.ch";
const char* WIFI_USERNAME = "bunleab.ch";
const char* WIFI_PASSWORD = "..."; 

// --- Cloud MQTT Broker Settings ---
const char* mqtt_server = "broker.hivemq.com"; 
const int mqtt_port = 1883;
const char* publish_topic = "rmuti/edge/bunleab/sensor1";

// --- Hardware Pins ---
#define RXD_PIN 4  // PMS7003 TX -> ESP32 RX
#define TXD_PIN 5  // PMS7003 RX -> ESP32 TX
#define I2C_SDA 20 // BME280 SDA
#define I2C_SCL 21 // BME280 SCL

// --- Global Objects & Variables ---
WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_BME280 bme; 

unsigned long lastMsg = 0;
const long interval = 60000; // Publish every 60 seconds
//const long interval = 300000; // Publish every 60 seconds
uint16_t latest_pm2_5 = 1;   // Store the most recent PM2.5 reading

// ══════════════════════════════════════════════════════════════
//  Wi-Fi Connection Function
// ══════════════════════════════════════════════════════════════
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("\n[WiFi] Connecting to %s", WIFI_SSID);
  
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);

  esp_eap_client_set_identity((uint8_t *)WIFI_IDENTITY, strlen(WIFI_IDENTITY));
  esp_eap_client_set_username((uint8_t *)WIFI_USERNAME, strlen(WIFI_USERNAME));
  esp_eap_client_set_password((uint8_t *)WIFI_PASSWORD, strlen(WIFI_PASSWORD));
  esp_wifi_sta_enterprise_enable();

  WiFi.begin(WIFI_SSID);

  unsigned long t = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - t > 20000) {            
      Serial.println("\n[WiFi] Timeout — will retry");
      return;
    }
  }
  Serial.printf("\n[WiFi] Connected  IP: %s\n", WiFi.localIP().toString().c_str());
}

// ══════════════════════════════════════════════════════════════
//  MQTT Reconnection Function
// ══════════════════════════════════════════════════════════════
void reconnect() {
  while (!client.connected() && WiFi.status() == WL_CONNECTED) {
    Serial.printf("[MQTT] Attempting connection to %s...", mqtt_server);
    
    String clientId = "AeroSense-ESP32-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("connected!");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// ══════════════════════════════════════════════════════════════
//  Setup
// ══════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(1500); 
  Serial.println("\n=== AeroSense Edge Node Starting ===");

  // 1. Initialize BME280
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!bme.begin(0x76, &Wire)) {
    Serial.println("[ERROR] Could not find a valid BME280 sensor!");
    while (1) { delay(10); } // Halt if BME fails
  }
  Serial.println("[Init] BME280 Initialized.");

  // 2. Initialize PMS7003
  Serial1.begin(9600, SERIAL_8N1, RXD_PIN, TXD_PIN);
  Serial.println("[Init] PMS7003 Initialized.");

  // 3. Connect Network & MQTT
  connectWiFi();
  client.setServer(mqtt_server, mqtt_port);
  Serial.println("====================================\n");
}

// ══════════════════════════════════════════════════════════════
//  Main Loop
// ══════════════════════════════════════════════════════════════
void loop() {
  // 1. Maintain Network Connections
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Lost connection — reconnecting...");
    connectWiFi();
  }
  
  if (!client.connected() && WiFi.status() == WL_CONNECTED) {
    reconnect();
  }
  
  client.loop(); // Keep MQTT alive

  // 2. Continuous Task: Read PMS7003 Buffer
  if (Serial1.available() >= 32) {
    uint8_t buffer[32];
    if (Serial1.peek() != 0x42) {
      Serial1.read(); 
    } else {
      Serial1.readBytes(buffer, 32);
      if (buffer[0] == 0x42 && buffer[1] == 0x4D) {
        latest_pm2_5 = (buffer[12] << 8) | buffer[13];
        if (latest_pm2_5 == 0) latest_pm2_5 = 1;
      }
    }
  }

  // 3. Timed Task: Read BME280 and Publish to Cloud
  unsigned long now = millis();
  if (now - lastMsg > interval) {
    lastMsg = now;

    // Read live data from BME280
    float temp = bme.readTemperature();
    float hum = bme.readHumidity();
    float pres = bme.readPressure() / 100.0F;

    // Format as JSON payload
    char jsonPayload[150];
    snprintf(jsonPayload, sizeof(jsonPayload), 
             "{\"pm25\":%u, \"temperature\":%.1f, \"humidity\":%.1f, \"pressure\":%.1f}", 
             latest_pm2_5, temp, hum, pres);

    Serial.print("[MQTT] Publishing live data: ");
    Serial.println(jsonPayload);
    
    // Send to Node-RED via HiveMQ
    client.publish(publish_topic, jsonPayload);
  }
  
  // Yield to background tasks
  delay(10);
}
