#include <SPI.h>

#define CS_1 10

#define TRANSDUCER_PIN 20

#define NUM_SAMPLES 32768

uint16_t values1[NUM_SAMPLES + 1];
uint16_t values2[NUM_SAMPLES + 1];
uint16_t   times[NUM_SAMPLES + 1];
char      output[NUM_SAMPLES + 1];

uint32_t startTime;

SPISettings settings(20000000, MSBFIRST,SPI_MODE2);

void clearSerial() {
  while(Serial.available() > 0) {
    Serial.readBytes(output, Serial.available());
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(3000);
  }
}  

void sample() {

  startTime = micros();

  for (int i = 0; i < NUM_SAMPLES; i++) {
    digitalWriteFast(TRANSDUCER_PIN, output[i]);
    times[i] = micros() - startTime;
    SPI.beginTransaction(settings);
    digitalWriteFast(CS_1, LOW);
    values1[i] = (SPI.transfer(0) << 8) + SPI.transfer(0);
    digitalWriteFast(CS_1, HIGH);
    SPI.endTransaction();
  }

  digitalWriteFast(TRANSDUCER_PIN, LOW);
  
}  

void send() {

  // Send the clock data
  for (int i = 0; i < NUM_SAMPLES; i++) {
    Serial.write(times[i] >> 8);
    Serial.write(times[i]);
  }
  
  // Send the data for mic 1
  for (int i = 0; i < NUM_SAMPLES; i++) {
    Serial.write(values1[i] >> 8);
    Serial.write(values1[i]);
  }

  /* // Send the data for mic 2 */
  /* for (int i = 0; i < NUM_SAMPLES; i++) { */
  /*   Serial.write(values2[i] >> 8); */
  /*   Serial.write(values2[i]); */
  /* } */
}

void slowFlash(int numTimes) {
  for (int i = 0; i < numTimes; i++) 
    digitalWrite(LED_BUILTIN, LOW);
    delay(150);{
    digitalWrite(LED_BUILTIN, HIGH);
    delay(300);
    digitalWrite(LED_BUILTIN, LOW);
    delay(150);
  }
}  

void fastFlash(int numTimes) {
  for (int i = 0; i < numTimes; i++) {
    digitalWrite(LED_BUILTIN, LOW);
    delay(50);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(50);
  }
}  

void setup() {

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(CS_1, OUTPUT);
  pinMode(TRANSDUCER_PIN, OUTPUT);

  delay(10);

  digitalWriteFast(TRANSDUCER_PIN, LOW);  
  digitalWriteFast(CS_1, HIGH);  

  delay(10);
  
  SPI.setSCK(27);
  SPI.begin();
  Serial.begin(19200);

  slowFlash(1);

  delay(5000);

  clearSerial();

  digitalWriteFast(LED_BUILTIN, HIGH);
}

void loop() {
  if (Serial.available() > 10){

    digitalWriteFast(LED_BUILTIN, LOW);
    
    Serial.readBytes(output, NUM_SAMPLES);
    
    sample();
    
    digitalWriteFast(LED_BUILTIN, HIGH);
    
    send();

  }
  delay(100);
}
