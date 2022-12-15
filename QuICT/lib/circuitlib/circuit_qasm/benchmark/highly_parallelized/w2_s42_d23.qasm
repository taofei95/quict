OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(0.4693616663132578) q[1];
rz(3.9783580456601815) q[0];
rz(4.643776189437103) q[0];
rz(5.698603192035316) q[1];
rz(1.4479402692620633) q[1];
rz(1.4101303353138346) q[0];
rz(1.129739320694302) q[0];
rz(1.7582942230837542) q[1];
rz(2.680548879175948) q[0];
rz(5.908178277053716) q[1];
rz(6.003799498904597) q[0];
rz(0.3875060947817857) q[1];
rz(1.6669225438584092) q[0];
rz(4.606623678131492) q[1];
rz(1.7049215092225996) q[1];
rz(4.148405826069329) q[0];
rz(1.7205788352648446) q[0];
rz(6.024614235022811) q[1];
cx q[1], q[0];
cx q[1], q[0];
rz(5.609363157042646) q[1];
rz(2.7366643941316156) q[0];
rz(0.07035314521330742) q[0];
rz(5.633888392779566) q[1];
rz(1.7226051881536688) q[0];
rz(1.0771772811635123) q[1];
cx q[0], q[1];
rz(5.778196618759672) q[1];
rz(4.526667903775494) q[0];
rz(1.775479043965405) q[0];
rz(1.157008268991553) q[1];
rz(1.5479938675469374) q[0];
rz(1.4105076459028658) q[1];
cx q[0], q[1];
rz(1.682739239499895) q[0];
rz(1.8607021968672537) q[1];
rz(2.707613318247816) q[1];
rz(4.498365477127802) q[0];
rz(3.4791901941007093) q[0];
rz(1.126506975120727) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
