OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
h q[2];
x q[12];
h q[5];
h q[9];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[2];
cx q[16], q[10];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
x q[6];
cx q[6], q[5];
cx q[15], q[18];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[3];
x q[17];
cx q[8], q[7];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
h q[2];
ry(1.5707963267948966) q[12];
x q[11];
rx(1.5707963267948966) q[15];
cx q[4], q[1];
rx(1.5707963267948966) q[18];
cx q[4], q[9];
rz(1.5707963267948966) q[9];
cx q[16], q[1];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[1];
cx q[5], q[10];
ry(1.5707963267948966) q[2];
x q[7];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[8];
cx q[13], q[5];
cx q[5], q[12];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
x q[15];
x q[17];
ry(1.5707963267948966) q[15];
x q[13];
x q[9];
ry(1.5707963267948966) q[16];
cx q[15], q[5];
cx q[6], q[5];
h q[17];
x q[6];
h q[2];
cx q[18], q[11];
h q[11];
cx q[8], q[15];
rx(1.5707963267948966) q[7];
h q[12];
ry(1.5707963267948966) q[10];
x q[17];
x q[6];
ry(1.5707963267948966) q[2];
x q[13];
rx(1.5707963267948966) q[4];
cx q[3], q[6];
h q[16];
rz(1.5707963267948966) q[7];
cx q[12], q[6];
rx(1.5707963267948966) q[17];
x q[4];
ry(1.5707963267948966) q[12];
x q[11];
cx q[18], q[5];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
x q[13];
x q[15];
rz(1.5707963267948966) q[12];
h q[15];
x q[18];
x q[9];
cx q[18], q[17];
rz(1.5707963267948966) q[15];
x q[3];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[1];
cx q[6], q[4];
h q[6];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
h q[2];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[12];
cx q[9], q[14];
cx q[12], q[11];
h q[10];
x q[15];
ry(1.5707963267948966) q[9];
cx q[4], q[1];
cx q[11], q[3];
x q[11];
x q[5];
h q[16];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[13];
h q[13];
rx(1.5707963267948966) q[10];
x q[6];
h q[18];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
x q[9];
h q[0];
rz(1.5707963267948966) q[4];
cx q[13], q[6];
cx q[10], q[0];
cx q[18], q[1];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[13];
cx q[3], q[4];
ry(1.5707963267948966) q[4];
x q[4];
x q[4];
x q[6];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[1];
x q[10];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[3];
x q[18];
ry(1.5707963267948966) q[8];
cx q[2], q[5];
cx q[8], q[2];
ry(1.5707963267948966) q[11];
x q[14];
ry(1.5707963267948966) q[2];
cx q[15], q[0];
rz(1.5707963267948966) q[16];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
x q[3];
rx(1.5707963267948966) q[10];
cx q[7], q[1];
rz(1.5707963267948966) q[3];
h q[12];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
cx q[2], q[14];
cx q[1], q[4];
rx(1.5707963267948966) q[1];
cx q[8], q[11];
x q[1];
cx q[8], q[7];
cx q[1], q[9];
rx(1.5707963267948966) q[18];
h q[8];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[8];
x q[17];
x q[11];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[14];
cx q[9], q[4];
h q[6];
rz(1.5707963267948966) q[16];
cx q[4], q[18];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[8];
x q[3];
rx(1.5707963267948966) q[7];
cx q[11], q[6];
h q[13];
h q[13];
x q[18];
cx q[3], q[7];
rz(1.5707963267948966) q[13];
cx q[12], q[6];
rx(1.5707963267948966) q[13];
h q[5];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[8];
h q[17];
h q[0];
rx(1.5707963267948966) q[3];
x q[18];
cx q[0], q[9];
rx(1.5707963267948966) q[7];
cx q[12], q[8];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[12];
cx q[14], q[6];
ry(1.5707963267948966) q[10];
x q[10];
cx q[14], q[5];
x q[10];
h q[13];
rz(1.5707963267948966) q[5];
h q[4];
h q[13];
rz(1.5707963267948966) q[2];
x q[7];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[15];
h q[1];
cx q[9], q[12];
cx q[9], q[4];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
x q[11];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
x q[18];
cx q[18], q[12];
x q[4];
rz(1.5707963267948966) q[17];
cx q[16], q[15];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[0];
h q[1];
x q[17];
cx q[16], q[15];
cx q[9], q[17];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
h q[9];
cx q[5], q[15];
h q[16];
cx q[18], q[10];
h q[17];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[9];
x q[18];
h q[13];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[17];
h q[18];
cx q[17], q[15];
ry(1.5707963267948966) q[6];
cx q[1], q[14];
h q[3];
cx q[13], q[6];
rx(1.5707963267948966) q[6];
cx q[0], q[13];
x q[13];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
x q[6];
h q[14];
x q[18];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[17];
x q[4];
cx q[14], q[18];
ry(1.5707963267948966) q[1];
x q[17];
rz(1.5707963267948966) q[0];
cx q[7], q[2];
rx(1.5707963267948966) q[14];
cx q[9], q[16];
cx q[15], q[13];
cx q[15], q[6];
ry(1.5707963267948966) q[16];
x q[3];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[1];
x q[5];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[9];
x q[8];
h q[14];
rz(1.5707963267948966) q[15];
cx q[11], q[3];
x q[1];
h q[11];
rz(1.5707963267948966) q[16];
cx q[1], q[6];
cx q[9], q[2];
rx(1.5707963267948966) q[3];
cx q[8], q[5];
ry(1.5707963267948966) q[5];
x q[14];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
h q[4];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[12];
cx q[18], q[14];
x q[5];
ry(1.5707963267948966) q[2];
x q[12];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[14];
cx q[11], q[12];
h q[4];
h q[9];
x q[8];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[17];
h q[11];
ry(1.5707963267948966) q[9];
cx q[5], q[4];
cx q[4], q[6];
cx q[16], q[5];
h q[10];
rz(1.5707963267948966) q[11];
cx q[12], q[0];
rx(1.5707963267948966) q[16];
h q[4];
x q[9];
x q[1];
x q[15];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
h q[11];
h q[0];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[8];
x q[18];
h q[8];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[10];
cx q[11], q[16];
x q[18];
rz(1.5707963267948966) q[4];
cx q[11], q[1];
h q[14];
rx(1.5707963267948966) q[3];
cx q[15], q[8];
rz(1.5707963267948966) q[2];
cx q[5], q[10];
cx q[10], q[18];
rx(1.5707963267948966) q[17];
h q[0];
h q[2];
x q[7];
cx q[16], q[12];
h q[6];
rz(1.5707963267948966) q[4];
x q[15];
cx q[18], q[3];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[10];
x q[18];
x q[1];
h q[6];
ry(1.5707963267948966) q[16];
h q[15];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[18];
cx q[7], q[6];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[11];
cx q[7], q[5];
x q[8];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[16];
cx q[1], q[7];
cx q[12], q[7];
cx q[11], q[3];
rx(1.5707963267948966) q[9];
cx q[10], q[14];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
h q[14];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
cx q[9], q[16];
cx q[15], q[2];
h q[12];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[9];
cx q[10], q[13];
rz(1.5707963267948966) q[10];
cx q[7], q[6];
x q[3];
rx(1.5707963267948966) q[3];
x q[6];
ry(1.5707963267948966) q[14];
h q[2];
cx q[14], q[8];
h q[4];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[1];
x q[14];
rz(1.5707963267948966) q[1];
x q[17];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[11];
x q[18];
cx q[16], q[1];
rz(1.5707963267948966) q[11];
x q[8];
ry(1.5707963267948966) q[6];
cx q[11], q[5];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[12];
h q[9];
cx q[8], q[18];
x q[6];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[16];
h q[13];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[5];
x q[15];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[12];
h q[13];
ry(1.5707963267948966) q[3];
cx q[4], q[15];
cx q[11], q[12];
cx q[1], q[2];
h q[2];
ry(1.5707963267948966) q[18];
h q[9];
x q[18];
cx q[12], q[10];
h q[17];
cx q[14], q[2];
h q[3];
rz(1.5707963267948966) q[10];
h q[18];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[17];
x q[6];
cx q[17], q[10];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[10];
cx q[2], q[4];
cx q[6], q[8];
h q[18];
h q[18];
cx q[10], q[17];
cx q[14], q[10];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[5];
cx q[5], q[7];
h q[1];
x q[0];
rz(1.5707963267948966) q[17];
cx q[16], q[8];
x q[13];
h q[11];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[15];
x q[0];
rz(1.5707963267948966) q[17];
h q[13];
rz(1.5707963267948966) q[0];
x q[9];
h q[3];
cx q[3], q[9];
h q[1];
rz(1.5707963267948966) q[6];
cx q[14], q[6];
h q[5];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[11];
x q[18];
h q[6];
ry(1.5707963267948966) q[17];
cx q[2], q[4];
h q[4];
h q[5];
cx q[11], q[3];
x q[7];
h q[0];
rz(1.5707963267948966) q[15];
x q[15];
h q[2];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
h q[10];
x q[7];
ry(1.5707963267948966) q[12];
h q[2];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[1];
h q[7];
ry(1.5707963267948966) q[1];
x q[0];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[16];
ry(1.5707963267948966) q[18];
x q[1];
cx q[17], q[3];
cx q[3], q[9];
rz(1.5707963267948966) q[2];
cx q[17], q[10];
cx q[7], q[5];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[18];
cx q[16], q[0];
cx q[8], q[1];
cx q[6], q[17];
cx q[5], q[14];
rz(1.5707963267948966) q[7];
cx q[0], q[3];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
x q[6];
rx(1.5707963267948966) q[4];
cx q[8], q[6];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[7];
x q[14];
h q[3];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[6];
cx q[7], q[8];
x q[6];
x q[10];
rz(1.5707963267948966) q[9];
h q[13];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
h q[4];
ry(1.5707963267948966) q[1];
cx q[14], q[6];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[13];
x q[2];
x q[17];
h q[17];
h q[12];
cx q[5], q[10];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[11];
x q[3];
cx q[1], q[2];
cx q[11], q[5];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[14];
cx q[1], q[12];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[8];
h q[11];
h q[6];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
cx q[15], q[3];
h q[18];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
h q[16];
cx q[1], q[4];
rz(1.5707963267948966) q[17];
cx q[0], q[16];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[10];
cx q[12], q[11];
rx(1.5707963267948966) q[0];
cx q[13], q[15];
h q[11];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[1];
h q[9];
ry(1.5707963267948966) q[14];
cx q[9], q[4];
cx q[16], q[13];
h q[5];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[8];
cx q[7], q[3];
h q[13];
cx q[12], q[15];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
cx q[0], q[3];
x q[17];
x q[12];
rx(1.5707963267948966) q[2];
x q[15];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[11];
h q[5];
rx(1.5707963267948966) q[0];
x q[4];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
cx q[0], q[15];
h q[15];
h q[12];
x q[16];
cx q[5], q[17];
x q[0];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[11];
h q[17];
cx q[1], q[16];
ry(1.5707963267948966) q[9];
cx q[5], q[14];
rz(1.5707963267948966) q[7];
x q[11];
rz(1.5707963267948966) q[7];
cx q[1], q[17];
x q[10];
x q[5];
cx q[1], q[4];
x q[5];
cx q[10], q[3];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[14];
cx q[7], q[17];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[11];
h q[0];
cx q[4], q[5];
cx q[8], q[7];
h q[5];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
x q[4];
cx q[3], q[4];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[0];
h q[4];
h q[13];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[7];
x q[11];
cx q[14], q[11];
x q[6];
ry(1.5707963267948966) q[16];
cx q[15], q[3];
x q[4];
rx(1.5707963267948966) q[17];
x q[13];
x q[1];
cx q[9], q[18];
rx(1.5707963267948966) q[17];
h q[1];
cx q[10], q[6];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[12];
cx q[16], q[11];
h q[8];
h q[6];
cx q[2], q[9];
cx q[16], q[18];
cx q[17], q[18];
cx q[13], q[2];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[4];
x q[14];
rx(1.5707963267948966) q[0];
cx q[12], q[16];
cx q[4], q[9];
rz(1.5707963267948966) q[16];
cx q[0], q[12];
cx q[5], q[7];
x q[13];
cx q[7], q[2];
rx(1.5707963267948966) q[3];
h q[9];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[3];
h q[15];
x q[10];
cx q[7], q[1];
rz(1.5707963267948966) q[9];
h q[0];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[3];
cx q[15], q[8];
h q[5];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[15];
x q[2];
ry(1.5707963267948966) q[0];
cx q[5], q[14];
cx q[11], q[9];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[0];
h q[16];
h q[17];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[17];
h q[9];
rx(1.5707963267948966) q[10];
x q[14];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
h q[13];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[5];
h q[4];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[13];
cx q[8], q[3];
h q[1];
x q[5];
x q[2];
rx(1.5707963267948966) q[2];
h q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[15];
h q[12];
cx q[17], q[11];
cx q[11], q[4];
x q[7];
h q[16];
ry(1.5707963267948966) q[15];
x q[17];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[13];
cx q[8], q[16];
h q[1];
rz(1.5707963267948966) q[5];
h q[7];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[1];
x q[3];
h q[8];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[1];
cx q[13], q[17];
rz(1.5707963267948966) q[9];
h q[5];
ry(1.5707963267948966) q[0];
cx q[4], q[13];
x q[1];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[8];
x q[0];
cx q[12], q[0];
h q[16];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[10];
h q[0];
cx q[4], q[15];
rx(1.5707963267948966) q[7];
h q[13];
cx q[6], q[13];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
x q[16];
h q[4];
x q[8];
x q[17];
cx q[16], q[6];
x q[9];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
h q[16];
h q[4];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[16];
ry(1.5707963267948966) q[12];
x q[11];
cx q[7], q[2];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[6];
cx q[16], q[3];
rx(1.5707963267948966) q[18];
h q[9];
rz(1.5707963267948966) q[7];
x q[13];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[15];
x q[7];
cx q[17], q[2];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[11];
x q[11];
rz(1.5707963267948966) q[2];
cx q[11], q[1];
rx(1.5707963267948966) q[5];
x q[3];
rz(1.5707963267948966) q[9];
cx q[12], q[2];
cx q[10], q[9];
x q[14];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
cx q[4], q[9];
h q[9];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[18];
cx q[4], q[8];
rz(1.5707963267948966) q[17];
cx q[13], q[8];
h q[15];
x q[1];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[2];
h q[0];
rx(1.5707963267948966) q[9];
x q[2];
h q[16];
cx q[13], q[14];
x q[2];
rx(1.5707963267948966) q[5];
h q[6];
x q[6];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[5];
x q[12];
ry(1.5707963267948966) q[2];
h q[11];
h q[0];
ry(1.5707963267948966) q[9];
cx q[14], q[3];
ry(1.5707963267948966) q[13];
x q[5];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[15];
x q[12];
rz(1.5707963267948966) q[9];
cx q[2], q[0];
x q[15];
x q[13];
x q[7];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
cx q[16], q[3];
h q[14];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[2];
cx q[1], q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[17];
x q[7];
cx q[13], q[3];
ry(1.5707963267948966) q[16];
x q[4];
h q[10];
h q[6];
h q[11];
ry(1.5707963267948966) q[1];
h q[6];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[8];
cx q[0], q[13];
rx(1.5707963267948966) q[16];
x q[9];
rx(1.5707963267948966) q[16];
h q[14];
h q[1];
cx q[12], q[18];
x q[9];
h q[14];
rz(1.5707963267948966) q[14];
h q[2];
ry(1.5707963267948966) q[7];
x q[2];
rx(1.5707963267948966) q[4];
x q[12];
rx(1.5707963267948966) q[17];
cx q[14], q[7];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[6];
cx q[10], q[0];
h q[8];
ry(1.5707963267948966) q[1];
h q[0];
rx(1.5707963267948966) q[13];
x q[10];
cx q[7], q[12];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[17];
h q[4];
rz(1.5707963267948966) q[16];
h q[8];
x q[12];
ry(1.5707963267948966) q[6];
cx q[2], q[8];
x q[15];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[12];
x q[1];
rx(1.5707963267948966) q[11];
x q[17];
x q[5];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
h q[17];
x q[0];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[11];
h q[14];
x q[0];
h q[17];
rz(1.5707963267948966) q[2];
h q[15];
x q[5];
h q[15];
x q[3];
cx q[5], q[16];
cx q[3], q[15];
cx q[13], q[14];
x q[14];
x q[8];
rx(1.5707963267948966) q[5];
x q[9];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[9];
cx q[17], q[16];
cx q[10], q[18];
h q[3];
x q[15];
rz(1.5707963267948966) q[17];
x q[15];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[17];
h q[6];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[17];
x q[6];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[13];
cx q[15], q[12];
ry(1.5707963267948966) q[6];
cx q[12], q[18];
rx(1.5707963267948966) q[16];
cx q[10], q[7];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
cx q[16], q[0];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[18];
cx q[15], q[14];
ry(1.5707963267948966) q[14];
x q[5];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[13];
h q[1];
cx q[0], q[13];
h q[2];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
cx q[1], q[13];
h q[3];
cx q[6], q[11];
cx q[6], q[16];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[17];
h q[0];
cx q[7], q[2];
cx q[0], q[6];
rz(1.5707963267948966) q[16];
h q[14];
ry(1.5707963267948966) q[17];
cx q[14], q[6];
cx q[7], q[4];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[11];
x q[0];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[15];
x q[15];
ry(1.5707963267948966) q[2];
h q[17];
cx q[10], q[16];
x q[15];
cx q[7], q[11];
rx(1.5707963267948966) q[5];
h q[15];
h q[9];
rz(1.5707963267948966) q[6];
cx q[13], q[16];
rz(1.5707963267948966) q[6];
x q[6];
rx(1.5707963267948966) q[6];
x q[17];
cx q[6], q[3];
h q[16];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
h q[16];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[8];
h q[9];
cx q[12], q[7];
x q[12];
x q[0];
h q[1];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
h q[9];
rx(1.5707963267948966) q[4];
cx q[6], q[3];
h q[18];
x q[13];
rx(1.5707963267948966) q[0];
h q[12];
h q[13];
h q[5];
x q[11];
cx q[13], q[0];
cx q[3], q[17];
cx q[13], q[17];
cx q[15], q[16];
x q[5];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
h q[9];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
cx q[17], q[9];
h q[7];
rz(1.5707963267948966) q[16];
h q[15];
cx q[0], q[18];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[18];
cx q[8], q[5];
rx(1.5707963267948966) q[6];
h q[16];
rx(1.5707963267948966) q[18];
x q[10];
rx(1.5707963267948966) q[8];
h q[1];
cx q[13], q[2];
cx q[0], q[8];
x q[15];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[18];
cx q[14], q[5];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[15];
cx q[15], q[5];
h q[9];
cx q[2], q[17];
ry(1.5707963267948966) q[0];
cx q[0], q[12];
x q[5];
rz(1.5707963267948966) q[1];
x q[14];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[17];
cx q[14], q[15];
ry(1.5707963267948966) q[10];
cx q[5], q[16];
x q[4];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[1];
h q[15];
x q[7];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[1];
x q[11];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[17];
x q[6];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
x q[16];
rx(1.5707963267948966) q[10];
h q[13];
cx q[16], q[0];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[16];
x q[2];
x q[4];
rx(1.5707963267948966) q[0];
cx q[6], q[7];
x q[1];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
cx q[0], q[14];
cx q[13], q[14];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
h q[7];
cx q[4], q[6];
cx q[3], q[13];
x q[15];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[7];
x q[4];
x q[8];
cx q[3], q[18];
ry(1.5707963267948966) q[8];
h q[14];
rx(1.5707963267948966) q[9];
cx q[5], q[8];
cx q[15], q[12];
h q[3];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[11];
cx q[13], q[7];
x q[5];
ry(1.5707963267948966) q[9];
x q[18];
cx q[15], q[10];
h q[9];
cx q[12], q[1];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[8];
x q[1];
h q[11];
rx(1.5707963267948966) q[2];
x q[3];
h q[16];
ry(1.5707963267948966) q[15];
cx q[7], q[4];
rx(1.5707963267948966) q[10];
cx q[0], q[4];
ry(1.5707963267948966) q[9];
x q[7];
cx q[17], q[7];
x q[15];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[12];
x q[16];
x q[4];
h q[3];
rx(1.5707963267948966) q[9];
cx q[6], q[5];
x q[11];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
h q[17];
x q[9];
rz(1.5707963267948966) q[7];
cx q[8], q[1];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[12];
cx q[8], q[13];
x q[0];
cx q[3], q[5];
x q[4];
ry(1.5707963267948966) q[4];
cx q[6], q[16];
ry(1.5707963267948966) q[14];
x q[11];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
cx q[2], q[5];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[17];
x q[10];
x q[15];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
cx q[13], q[3];
cx q[14], q[9];
rx(1.5707963267948966) q[17];
cx q[10], q[5];
x q[16];
rx(1.5707963267948966) q[17];
h q[18];
x q[13];
h q[12];
rz(1.5707963267948966) q[9];
cx q[2], q[18];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[11];
x q[13];
cx q[13], q[1];
x q[3];
cx q[4], q[17];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[18];
h q[6];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[16];
h q[12];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[8];
cx q[14], q[11];
h q[2];
h q[16];
x q[0];
cx q[1], q[7];
ry(1.5707963267948966) q[3];
h q[8];
h q[10];
cx q[13], q[12];
cx q[9], q[2];
cx q[8], q[13];
ry(1.5707963267948966) q[1];
x q[14];
cx q[10], q[7];
x q[14];
rz(1.5707963267948966) q[11];
x q[4];
h q[10];
rz(1.5707963267948966) q[6];
h q[4];
cx q[7], q[16];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[5];
cx q[10], q[13];
h q[0];
x q[18];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[18];
h q[1];
x q[3];
h q[8];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[6];
x q[0];
cx q[9], q[1];
h q[2];
h q[11];
h q[11];
h q[6];
rx(1.5707963267948966) q[7];
cx q[16], q[3];
x q[4];
rx(1.5707963267948966) q[18];
cx q[16], q[18];
cx q[17], q[14];
rz(1.5707963267948966) q[7];
x q[8];
cx q[12], q[10];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[13];
cx q[10], q[17];
cx q[1], q[8];
rx(1.5707963267948966) q[9];
cx q[15], q[9];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[1];
x q[14];
x q[0];
cx q[12], q[5];
x q[2];
h q[18];
rz(1.5707963267948966) q[2];
h q[11];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
cx q[15], q[17];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[10];
cx q[15], q[2];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[16];
x q[16];
ry(1.5707963267948966) q[7];
h q[18];
rx(1.5707963267948966) q[9];
cx q[16], q[18];
rx(1.5707963267948966) q[2];
h q[0];
x q[5];
x q[3];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[6];
cx q[10], q[18];
x q[17];
x q[12];
cx q[18], q[1];
h q[0];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[12];
h q[3];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[15];
x q[3];
cx q[15], q[14];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[13];
h q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
x q[16];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
h q[1];
ry(1.5707963267948966) q[14];
cx q[9], q[13];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[6];
cx q[5], q[14];
cx q[3], q[18];
cx q[0], q[7];
rx(1.5707963267948966) q[4];
x q[7];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
h q[4];
h q[2];
x q[7];
x q[4];
h q[14];
ry(1.5707963267948966) q[15];
x q[4];
x q[14];
x q[2];
h q[5];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[18];
h q[18];
x q[16];
h q[16];
h q[6];
x q[13];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[9];
cx q[15], q[9];
h q[10];
rx(1.5707963267948966) q[5];
x q[1];
cx q[5], q[2];
ry(1.5707963267948966) q[3];
h q[17];
h q[2];
x q[12];
ry(1.5707963267948966) q[18];
x q[14];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[15];
cx q[18], q[4];
ry(1.5707963267948966) q[6];
h q[12];
x q[3];
x q[9];
h q[16];
cx q[11], q[18];
rx(1.5707963267948966) q[12];
cx q[15], q[14];
rx(1.5707963267948966) q[15];
h q[15];
x q[11];
cx q[13], q[15];
ry(1.5707963267948966) q[16];
cx q[16], q[9];
rz(1.5707963267948966) q[14];
h q[1];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[15];
cx q[8], q[9];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[9];
h q[3];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
cx q[10], q[1];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[15];
x q[7];
rx(1.5707963267948966) q[10];
cx q[8], q[5];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[16];
h q[15];
x q[0];
ry(1.5707963267948966) q[18];
h q[10];
cx q[0], q[1];
h q[5];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[11];
cx q[6], q[13];
cx q[2], q[12];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[10];
