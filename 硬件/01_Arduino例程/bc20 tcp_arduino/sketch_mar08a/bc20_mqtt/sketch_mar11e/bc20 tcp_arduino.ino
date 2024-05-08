/*
  Blink
  Turns on an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the Uno and
  Leonardo, it is attached to digital pin 13. If you're unsure what
  pin the on-board LED is connected to on your Arduino model, check
  the documentation at http://arduino.cc

  This example code is in the public domain.

  modified 8 May 2014
  by Scott Fitzgerald
 */
//判断返回值是否正确
void connectGSM (String cmd, char *res)
{
  while(1)
  {
    Serial.println(cmd);
    delay(500);
    while(Serial.available()>0)
    {
      
      if(Serial.find(res))
      {
  digitalWrite(13, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(100);              // wait for a second
  digitalWrite(13, LOW);    // turn the LED off by making the voltage LOW
  delay(100);     
        return;
      }
    }
    delay(1000);
   }
}

// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin 13 as an output.
  pinMode(13, OUTPUT);
    Serial.begin(115200); //
  while (!Serial) {
    ; // wait for serial port to connect. Needed for Leonardo only
  }
    // prints title with ending line break 
   connectGSM("AT","OK");
   Serial.println("ATE0&W");//关闭回显
   delay(300);  
   connectGSM("AT+CPIN?","+CPIN: READY");//返+CPIN:READY，表明识别到卡了
   connectGSM("AT+CGATT?","+CGATT: 1");//返+CGACT: 1,就能正常工作了
   Serial.println("AT+QICLOSE=0");//关闭上一次socekt连接
   delay(300);  
   connectGSM("AT+QIOPEN=1,0,\"TCP\",\"114.115.148.172\",10000,1234,0","+QIOPEN: 0,0");//建立服务器的IP和端口连接
}

// the loop function runs over and over again forever
void loop() {
  digitalWrite(13, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);              // wait for a second
  digitalWrite(13, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);              // wait for a second
      // prints title with ending line break 
   connectGSM("AT+QISEND=0,10,9876543210","SEND OK");
}
/*
  Blink
  Turns on an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the Uno and
  Leonardo, it is attached to digital pin 13. If you're unsure what
  pin the on-board LED is connected to on your Arduino model, check
  the documentation at http://arduino.cc

  This example code is in the public domain.

  modified 8 May 2014
  by Scott Fitzgerald
 */



