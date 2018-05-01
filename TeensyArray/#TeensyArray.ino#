#include <ADC.h>

#define PIN_1 A9
#define PIN_2 A3

#define HIGH_TRANSDUCER_PIN 22
#define LOW_TRANSDUCER_PIN 16

#define RESOLUTION 12

ADC *adc = new ADC();

uint16_t numSamples = 8192;

uint16_t values1[8193];
uint16_t values2[8193];
char output[1025];

uint32_t startTime;

void clearSerial() {
  while(Serial.available() > 0) {
    Serial.readBytes(output, Serial.available());
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(2000);
  }
}  

void sample() {

  startTime = micros();

  for (int i = 0; (i < numSamples) && (i < 8192); i++) {
    if ((output[i/8] >> (i % 8)) & 1) {
      digitalWriteFast(LOW_TRANSDUCER_PIN, LOW);
      digitalWriteFast(HIGH_TRANSDUCER_PIN, HIGH);
    } else {
      digitalWriteFast(HIGH_TRANSDUCER_PIN, LOW);
      digitalWriteFast(LOW_TRANSDUCER_PIN, HIGH);
    }
    values1[i] = adc->adc0->analogRead(PIN_1);
    values2[i] = adc->adc1->analogRead(PIN_2);
  }

  startTime = micros() - startTime;

  digitalWriteFast(LOW_TRANSDUCER_PIN, LOW);
  digitalWriteFast(HIGH_TRANSDUCER_PIN, LOW); 
}  

void send() {

  // Write the number of samples we took
  Serial.write(numSamples >> 8);
  Serial.write(numSamples);

  // Write the time it took us to take those samples
  Serial.write(startTime >> 24);
  Serial.write(startTime >> 16);
  Serial.write(startTime >> 8);
  Serial.write(startTime);
  
  // Send the data for mic 1
  for (int i = 0; i < numSamples; i++) {
    Serial.write(values1[i] >> 8);
    Serial.write(values1[i]);
  }

  // Send the data for mic 2
  for (int i = 0; i < numSamples; i++) {
    Serial.write(values2[i] >> 8);
    Serial.write(values2[i]);
  }
}

void flash(int numTimes) {
  for (int i = 0; i < numTimes; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(100);
  }
}  

void setup() {

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LOW_TRANSDUCER_PIN, OUTPUT);
  pinMode(HIGH_TRANSDUCER_PIN, OUTPUT);
  pinMode(PIN_1, INPUT);
  pinMode(PIN_2, INPUT);
  
  adc->setAveraging(1); // set number of averages
  adc->setResolution(RESOLUTION); // set bits of resolution
  adc->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
  adc->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
  
  adc->setAveraging(1, ADC_1); // set number of averages
  adc->setResolution(RESOLUTION, ADC_1); // set bits of resolution
  adc->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED, ADC_1); // change the conversion speed
  adc->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED, ADC_1); // change the sampling speed

  adc->adc0->analogRead(PIN_1);
  adc->adc1->analogRead(PIN_2);

  Serial.begin(19200);
  delay(2000); // Give the Serial time to start

  clearSerial();

  digitalWriteFast(LOW_TRANSDUCER_PIN, LOW);
  digitalWriteFast(HIGH_TRANSDUCER_PIN, LOW);
  
  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  if (Serial.available() > 1){

    byte msb = Serial.read();
    byte lsb = Serial.read();
    numSamples = (msb << 8) + lsb;
     
    digitalWrite(LED_BUILTIN, LOW);

    Serial.readBytes(output, (int) (numSamples / 8));

    sample();

    digitalWrite(LED_BUILTIN, HIGH);
    send();
    
  } else {
    sample();
  }
}
