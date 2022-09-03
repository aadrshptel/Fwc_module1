#include <Arduino.h>
int Z,Y,X,W;
int A,B,C,D,CLK;
void disp_7474(int D,int C,int B,int A)
{
 digitalWrite(2, A);
 digitalWrite(3, B);
 digitalWrite(4, C);
 digitalWrite(5, D);
}

void setup()
{
    pinMode(2, OUTPUT);
    pinMode(3, OUTPUT);
    pinMode(4, OUTPUT);
    pinMode(5, OUTPUT);
    pinMode(13,OUTPUT);
    pinMode(6, INPUT);
    pinMode(7, INPUT);
    pinMode(8, INPUT);
    pinMode(9, INPUT);
}
void loop()
{
 digitalWrite(13, HIGH);
 delay(500);
 W = digitalRead(6);
 X = digitalRead(7);
 Y = digitalRead(8);
 Z = digitalRead(9);
 A = (!Z&&!W) || (!Y&&!X&&!W);
 B = (W&&!X&&!Y&&!Z) || (!W&&X&&!Y&&!Z) || (W&&!X&&Y&&!Z) || (!W&&X&&Y&&!Z);
 C = (W&&X&&!Y&&!Z) || (!W&&!X&&Y&&!Z) || (W&&!X&&Y&&!Z) || (!W&&X&&Y&&!Z);
 D = (W&&X&&Y&&!Z)||(!W&&!X&&!Y&&Z);
 disp_7474(D,C,B,A);
 digitalWrite(13, LOW);
 delay(500);
}
