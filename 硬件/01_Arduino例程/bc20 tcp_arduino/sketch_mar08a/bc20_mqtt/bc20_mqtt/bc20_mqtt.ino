#include <SoftwareSerial.h>

SoftwareSerial mySerial(10, 11); // RX, TX

void connectNBIOT (String cmd, char *res)
{
  while(1)
  {
    mySerial.println(cmd);
    delay(500);
    while(mySerial.available()>0)
    {
      if(mySerial.find(res))
      {
  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(100);              // wait for a second
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(100);     
        return;
      }
    }
    delay(1000);
   }
}

void setup() {
  // initialize digital pin 13 as an output.
  pinMode(13, OUTPUT);
    Serial.begin(115200);
    mySerial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for Leonardo only
  }
    // prints title with ending line break 
   connectNBIOT("AT","OK");
   mySerial.println("ATE0");//关闭回显
   delay(300);  
   connectNBIOT("AT+CPIN?","+CPIN: READY");//返+CPIN:READY，表明识别到卡了
   connectNBIOT("AT+CGATT?","+CGATT: 1");//返+CGACT: 1,就能正常工作了
   mySerial.println("AT+QMTCLOSE=0");//关闭上一次socekt连接
   delay(300);
   connectNBIOT("AT+QMTCFG=\"aliauth\",0,\"hh83tsRhk7S\",\"e0i3H75WfbubjgfmHJdb\",\"f819d8c3032d63376649499af\"","OK");
   delay(300);  
   connectNBIOT("AT+QMTOPEN=0,\"iot-as-mqtt.cn-shanghai.aliyuncs.com\",1883","+QMTOPEN: 0,0");//建立服务器的IP和端口连接
   delay(300);  
   connectNBIOT("AT+QMTCONN=0,\"e0i3H75WfbubjgfmHJdb\"","+QMTCONN: 0,0,0");//建立服务器的IP和端口连接
    delay(300);     
}

// the loop function runs over and over again forever
void loop() {
  digitalWrite(13, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);              // wait for a second
  digitalWrite(13, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);              // wait for a second
      // prints title with ending line break 
   connectNBIOT("AT+QMTPUB=0,0,0,0,\"topic/pub\",\"hello MQTT.\"","+QMTPUB: 0,0,0");//send message
}
