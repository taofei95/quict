OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
h q[19];
rz(1.5707963267948966) q[22];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[19];
h q[12];
h q[16];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[8];
cx q[19], q[18];
cx q[0], q[3];
ry(1.5707963267948966) q[10];
cx q[9], q[26];
rz(1.5707963267948966) q[3];
h q[17];
ry(1.5707963267948966) q[26];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[23];
h q[20];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[0];
h q[24];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[13];
cx q[19], q[22];
ry(1.5707963267948966) q[0];
cx q[5], q[1];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[17];
h q[2];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
cx q[24], q[18];
h q[2];
rx(1.5707963267948966) q[2];
h q[12];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[16];
cx q[1], q[15];
rx(1.5707963267948966) q[14];
h q[1];
cx q[20], q[1];
cx q[16], q[9];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[18];
cx q[1], q[23];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[21];
cx q[19], q[13];
h q[0];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[11];
h q[12];
h q[13];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[12];
cx q[8], q[4];
rx(1.5707963267948966) q[23];
h q[25];
rz(1.5707963267948966) q[1];
h q[16];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[7];
h q[17];
rz(1.5707963267948966) q[23];
h q[21];
rx(1.5707963267948966) q[12];
h q[16];
rx(1.5707963267948966) q[15];
cx q[13], q[11];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[14];
cx q[25], q[4];
cx q[4], q[12];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[20];
cx q[18], q[19];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[1];
h q[2];
cx q[26], q[0];
ry(1.5707963267948966) q[26];
h q[26];
ry(1.5707963267948966) q[21];
cx q[1], q[17];
rx(1.5707963267948966) q[19];
h q[23];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[1];
cx q[1], q[11];
h q[20];
rz(1.5707963267948966) q[22];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[19];
cx q[4], q[13];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[13];
cx q[22], q[21];
ry(1.5707963267948966) q[20];
cx q[20], q[26];
rz(1.5707963267948966) q[26];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[11];
h q[11];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[23];
ry(1.5707963267948966) q[5];
h q[17];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
h q[24];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[11];
cx q[21], q[15];
rx(1.5707963267948966) q[16];
cx q[19], q[7];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[4];
cx q[17], q[4];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[25];
cx q[8], q[20];
h q[6];
cx q[10], q[13];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[20];
cx q[6], q[10];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[23];
cx q[26], q[2];
h q[10];
cx q[6], q[1];
rz(1.5707963267948966) q[18];
h q[7];
rx(1.5707963267948966) q[4];
cx q[15], q[2];
cx q[14], q[11];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[16];
h q[16];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[19];
cx q[5], q[2];
ry(1.5707963267948966) q[3];
cx q[14], q[13];
cx q[4], q[20];
rx(1.5707963267948966) q[3];
h q[10];
rz(1.5707963267948966) q[23];
h q[5];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[21];
cx q[17], q[18];
h q[21];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[18];
cx q[13], q[15];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[26];
h q[10];
h q[17];
rz(1.5707963267948966) q[25];
h q[1];
cx q[0], q[10];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[21];
cx q[20], q[6];
cx q[9], q[8];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[8];
cx q[8], q[1];
h q[14];
cx q[16], q[0];
cx q[11], q[15];
cx q[0], q[6];
rx(1.5707963267948966) q[23];
cx q[10], q[19];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[25];
rz(1.5707963267948966) q[21];
h q[6];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
h q[12];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
h q[18];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[16];
cx q[19], q[7];
rx(1.5707963267948966) q[22];
rz(1.5707963267948966) q[13];
h q[4];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[5];
cx q[16], q[4];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[15];
h q[9];
rz(1.5707963267948966) q[8];
cx q[18], q[9];
rz(1.5707963267948966) q[22];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[2];
cx q[6], q[26];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[8];
cx q[22], q[15];
h q[2];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[11];
h q[2];
h q[14];
rz(1.5707963267948966) q[19];
h q[9];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[11];
h q[18];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[10];
h q[25];
cx q[13], q[2];
cx q[9], q[6];
cx q[10], q[3];
ry(1.5707963267948966) q[26];
cx q[9], q[7];
cx q[24], q[15];
rz(1.5707963267948966) q[3];
h q[13];
h q[1];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[7];
h q[23];
cx q[1], q[10];
cx q[4], q[10];
cx q[17], q[0];
rx(1.5707963267948966) q[3];
h q[8];
cx q[2], q[8];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[21];
cx q[0], q[12];
rx(1.5707963267948966) q[10];
h q[8];
ry(1.5707963267948966) q[4];
h q[14];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
cx q[9], q[14];
h q[3];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[7];
cx q[13], q[11];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
h q[0];
rz(1.5707963267948966) q[9];
h q[1];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[9];
h q[10];
h q[21];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[16];
cx q[8], q[6];
h q[23];
rx(1.5707963267948966) q[24];
cx q[6], q[19];
h q[24];
cx q[25], q[8];
ry(1.5707963267948966) q[2];
h q[14];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[16];
cx q[7], q[25];
cx q[24], q[2];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[6];
h q[7];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[22];
cx q[22], q[13];
rz(1.5707963267948966) q[25];
cx q[12], q[15];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[18];
h q[26];
h q[9];
h q[1];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[11];
h q[2];
ry(1.5707963267948966) q[7];
cx q[26], q[20];
cx q[11], q[19];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
h q[8];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[15];
h q[8];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[11];
h q[6];
h q[10];
cx q[17], q[9];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[21];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[6];
