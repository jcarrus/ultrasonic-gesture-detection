#include <ADC.h>

#define PIN_1 A9
#define PIN_2 A3
#define READ_RESOLUTION 12

#define TRANSDUCER_PIN 20
#define WRITE_RESOLUTION 8

#define NUM_SAMPLES 8192

ADC *adc = new ADC();

uint16_t values1[NUM_SAMPLES + 1];
uint16_t values2[NUM_SAMPLES + 1];
uint16_t   times[NUM_SAMPLES + 1];
char      output[NUM_SAMPLES + 1];

uint32_t startTime;

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
    analogWrite(TRANSDUCER_PIN, output[i]);
    values1[i] = adc->adc0->analogRead(PIN_1);
    values2[i] = adc->adc1->analogRead(PIN_2);
    times[i] = micros() - startTime;
  }

  analogWrite(TRANSDUCER_PIN, 0);
  
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

  // Send the data for mic 2
  for (int i = 0; i < NUM_SAMPLES; i++) {
    Serial.write(values2[i] >> 8);
    Serial.write(values2[i]);
  }
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
  pinMode(TRANSDUCER_PIN, OUTPUT);

  delay(10);

  analogWriteFrequency(TRANSDUCER_PIN, 234375);
  analogWriteResolution(WRITE_RESOLUTION);
  pinMode(PIN_1, INPUT);
  pinMode(PIN_2, INPUT);

  delay(10);
  
  adc->setAveraging(1); // set number of averages
  adc->setResolution(READ_RESOLUTION); // set bits of resolution
  adc->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
  adc->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
  adc->setAveraging(1, ADC_1); // set number of averages
  adc->setResolution(READ_RESOLUTION, ADC_1); // set bits of resolution
  adc->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED, ADC_1); // change the conversion speed
  adc->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED, ADC_1); // change the sampling speed
  adc->adc0->analogRead(PIN_1);
  adc->adc1->analogRead(PIN_2);
  analogWrite(TRANSDUCER_PIN, 0);  

  delay(10);

  Serial.begin(19200);

  slowFlash(1);

  delay(5000);

  clearSerial();

  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  if (Serial.available() > 10){

    digitalWrite(LED_BUILTIN, LOW);
    
    Serial.readBytes(output, NUM_SAMPLES);
    
    sample();
    
    digitalWrite(LED_BUILTIN, HIGH);
    
    send();

  }
  delay(100);
}
