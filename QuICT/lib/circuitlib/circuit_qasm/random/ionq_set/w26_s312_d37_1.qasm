OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
rx(1.5707963267948966) q[5];
rxx(0) q[4], q[19];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[24];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[2];
rxx(0) q[10], q[7];
rxx(0) q[2], q[11];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[15];
rxx(0) q[11], q[14];
rxx(0) q[11], q[23];
rx(1.5707963267948966) q[23];
rxx(0) q[21], q[10];
ry(1.5707963267948966) q[0];
rxx(0) q[14], q[12];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[4];
rxx(0) q[1], q[19];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[25];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
rxx(0) q[11], q[13];
ry(1.5707963267948966) q[19];
rxx(0) q[6], q[23];
rxx(0) q[16], q[5];
rx(1.5707963267948966) q[23];
rxx(0) q[0], q[17];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[3];
rxx(0) q[3], q[11];
ry(1.5707963267948966) q[24];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
rxx(0) q[0], q[9];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
rxx(0) q[11], q[24];
rx(1.5707963267948966) q[9];
rxx(0) q[14], q[7];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[23];
ry(1.5707963267948966) q[19];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rxx(0) q[23], q[6];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[8];
rxx(0) q[23], q[19];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[19];
rxx(0) q[10], q[25];
rz(1.5707963267948966) q[19];
rxx(0) q[0], q[5];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[15];
rxx(0) q[15], q[24];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[16];
rxx(0) q[1], q[2];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
rxx(0) q[3], q[18];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[25];
rxx(0) q[19], q[5];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
rxx(0) q[19], q[11];
rz(1.5707963267948966) q[22];
rxx(0) q[22], q[1];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[25];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[25];
rxx(0) q[23], q[2];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[9];
rxx(0) q[20], q[2];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[25];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[17];
rxx(0) q[15], q[19];
rxx(0) q[6], q[1];
rxx(0) q[7], q[15];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[3];
rxx(0) q[1], q[12];
rxx(0) q[18], q[3];
rxx(0) q[8], q[10];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[23];
rxx(0) q[22], q[12];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[14];
rxx(0) q[13], q[2];
ry(1.5707963267948966) q[19];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[15];
rxx(0) q[7], q[2];
rx(1.5707963267948966) q[25];
rz(1.5707963267948966) q[15];
rxx(0) q[24], q[2];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[14];
rxx(0) q[17], q[9];
rxx(0) q[20], q[4];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[25];
rxx(0) q[7], q[5];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[10];
rxx(0) q[6], q[22];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[25];
rxx(0) q[20], q[12];
rxx(0) q[11], q[6];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[22];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[22];
rxx(0) q[24], q[6];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
rxx(0) q[2], q[8];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[12];
rxx(0) q[15], q[1];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[19];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[12];
rxx(0) q[4], q[14];
rxx(0) q[0], q[1];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[17];
rxx(0) q[7], q[21];
ry(1.5707963267948966) q[9];
rxx(0) q[13], q[20];
rxx(0) q[17], q[8];
ry(1.5707963267948966) q[19];
rxx(0) q[7], q[21];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[15];
rxx(0) q[1], q[18];
ry(1.5707963267948966) q[25];
rxx(0) q[6], q[20];
rz(1.5707963267948966) q[15];
rxx(0) q[20], q[17];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[18];
rxx(0) q[20], q[11];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[6];
rxx(0) q[13], q[14];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[12];
rxx(0) q[12], q[5];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[2];
rxx(0) q[1], q[16];
rx(1.5707963267948966) q[24];
rxx(0) q[2], q[8];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[10];
rxx(0) q[21], q[18];
rxx(0) q[7], q[13];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[4];
rxx(0) q[14], q[18];
rz(1.5707963267948966) q[11];
rxx(0) q[1], q[8];
rxx(0) q[20], q[5];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[6];
