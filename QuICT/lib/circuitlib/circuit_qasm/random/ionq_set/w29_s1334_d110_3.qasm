OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[5];
rxx(0) q[6], q[23];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
rxx(0) q[5], q[27];
ry(1.5707963267948966) q[28];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[23];
rxx(0) q[23], q[13];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[15];
rxx(0) q[24], q[10];
ry(1.5707963267948966) q[14];
rxx(0) q[0], q[12];
rz(1.5707963267948966) q[27];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[22];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[1];
rxx(0) q[13], q[18];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rxx(0) q[27], q[7];
rxx(0) q[15], q[24];
rx(1.5707963267948966) q[26];
rz(1.5707963267948966) q[24];
rxx(0) q[23], q[18];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
rxx(0) q[11], q[20];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[13];
rxx(0) q[6], q[3];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[17];
rxx(0) q[8], q[22];
rxx(0) q[23], q[15];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[27];
rx(1.5707963267948966) q[26];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[27];
ry(1.5707963267948966) q[3];
rxx(0) q[11], q[22];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[15];
rxx(0) q[5], q[11];
ry(1.5707963267948966) q[25];
rxx(0) q[21], q[0];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[18];
rxx(0) q[4], q[28];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[16];
rx(1.5707963267948966) q[26];
rx(1.5707963267948966) q[8];
rxx(0) q[3], q[0];
rxx(0) q[15], q[21];
rxx(0) q[14], q[5];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[24];
rxx(0) q[11], q[15];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[26];
rxx(0) q[21], q[14];
rz(1.5707963267948966) q[22];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
rxx(0) q[11], q[27];
rxx(0) q[17], q[10];
rz(1.5707963267948966) q[16];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[24];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[26];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[4];
rxx(0) q[7], q[15];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[28];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[20];
rxx(0) q[23], q[16];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[23];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[28];
rxx(0) q[4], q[10];
rz(1.5707963267948966) q[28];
ry(1.5707963267948966) q[14];
rxx(0) q[18], q[19];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[26];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
rxx(0) q[26], q[21];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[22];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[20];
rxx(0) q[21], q[9];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
rxx(0) q[27], q[22];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[21];
rxx(0) q[19], q[28];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[8];
rxx(0) q[5], q[28];
rz(1.5707963267948966) q[3];
rxx(0) q[23], q[11];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[19];
rz(1.5707963267948966) q[16];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[16];
rxx(0) q[21], q[26];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[21];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[27];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[28];
rx(1.5707963267948966) q[23];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[25];
rxx(0) q[12], q[19];
ry(1.5707963267948966) q[28];
rz(1.5707963267948966) q[4];
rxx(0) q[2], q[17];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[2];
rxx(0) q[27], q[18];
rz(1.5707963267948966) q[24];
rxx(0) q[27], q[20];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[28];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[1];
rxx(0) q[18], q[22];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[19];
rxx(0) q[20], q[16];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[15];
rxx(0) q[28], q[1];
rxx(0) q[12], q[8];
rx(1.5707963267948966) q[26];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[24];
rxx(0) q[21], q[25];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[23];
rxx(0) q[25], q[6];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[25];
rxx(0) q[15], q[19];
ry(1.5707963267948966) q[26];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[1];
rxx(0) q[3], q[16];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[15];
rxx(0) q[14], q[15];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[27];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[19];
rxx(0) q[4], q[23];
rx(1.5707963267948966) q[24];
rxx(0) q[8], q[20];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[19];
rxx(0) q[23], q[3];
rxx(0) q[15], q[26];
rz(1.5707963267948966) q[4];
rxx(0) q[20], q[28];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[0];
rxx(0) q[9], q[24];
rz(1.5707963267948966) q[20];
rxx(0) q[24], q[1];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
rxx(0) q[23], q[28];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[26];
rxx(0) q[14], q[28];
ry(1.5707963267948966) q[7];
rxx(0) q[20], q[4];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[9];
rxx(0) q[20], q[19];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[16];
rxx(0) q[18], q[14];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[6];
rxx(0) q[3], q[13];
rz(1.5707963267948966) q[18];
rxx(0) q[2], q[22];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[28];
rx(1.5707963267948966) q[16];
rxx(0) q[11], q[5];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[22];
rz(1.5707963267948966) q[8];
rxx(0) q[10], q[20];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[13];
rxx(0) q[21], q[28];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[5];
rxx(0) q[17], q[14];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rxx(0) q[20], q[6];
rx(1.5707963267948966) q[14];
rxx(0) q[16], q[1];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[23];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[1];
rxx(0) q[16], q[7];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[25];
rxx(0) q[24], q[9];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[23];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[0];
rxx(0) q[27], q[24];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[18];
rxx(0) q[26], q[8];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[20];
rxx(0) q[18], q[6];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[17];
rz(1.5707963267948966) q[9];
rxx(0) q[4], q[28];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[22];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[27];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[11];
rxx(0) q[27], q[20];
rxx(0) q[0], q[18];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[28];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
rxx(0) q[2], q[23];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[22];
rxx(0) q[2], q[5];
rx(1.5707963267948966) q[23];
rxx(0) q[2], q[26];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[22];
ry(1.5707963267948966) q[28];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[27];
rz(1.5707963267948966) q[18];
rxx(0) q[3], q[24];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[23];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[2];
rxx(0) q[18], q[27];
rz(1.5707963267948966) q[26];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[25];
rx(1.5707963267948966) q[25];
rz(1.5707963267948966) q[28];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[2];
rxx(0) q[13], q[12];
rx(1.5707963267948966) q[28];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[26];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[27];
rxx(0) q[15], q[27];
rz(1.5707963267948966) q[28];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[15];
rx(1.5707963267948966) q[26];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[24];
rxx(0) q[11], q[5];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[23];
rxx(0) q[10], q[14];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[21];
rxx(0) q[21], q[13];
rx(1.5707963267948966) q[28];
ry(1.5707963267948966) q[28];
rxx(0) q[16], q[27];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[26];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[0];
rxx(0) q[16], q[14];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[8];
rxx(0) q[23], q[7];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[28];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[25];
rxx(0) q[14], q[0];
ry(1.5707963267948966) q[11];
rxx(0) q[3], q[2];
ry(1.5707963267948966) q[2];
rxx(0) q[0], q[9];
rxx(0) q[14], q[8];
rxx(0) q[17], q[5];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[7];
rxx(0) q[25], q[18];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[6];
rxx(0) q[15], q[26];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[27];
rxx(0) q[2], q[27];
rx(1.5707963267948966) q[2];
rxx(0) q[4], q[5];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[13];
rxx(0) q[26], q[13];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[27];
ry(1.5707963267948966) q[7];
rxx(0) q[20], q[24];
rxx(0) q[24], q[3];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[28];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[26];
rxx(0) q[17], q[10];
rxx(0) q[0], q[18];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[15];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[11];
rxx(0) q[20], q[2];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[25];
rxx(0) q[17], q[9];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rxx(0) q[6], q[27];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[23];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[18];
rxx(0) q[25], q[16];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[13];
rz(1.5707963267948966) q[11];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[5];
rxx(0) q[5], q[27];
rxx(0) q[0], q[21];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[22];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[12];
rxx(0) q[28], q[9];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[25];
ry(1.5707963267948966) q[26];
ry(1.5707963267948966) q[11];
rxx(0) q[23], q[15];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[12];
rxx(0) q[26], q[2];
ry(1.5707963267948966) q[18];
ry(1.5707963267948966) q[26];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[28];
rxx(0) q[22], q[9];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[22];
rxx(0) q[24], q[26];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[19];
rxx(0) q[14], q[2];
ry(1.5707963267948966) q[10];
rxx(0) q[19], q[12];
rx(1.5707963267948966) q[13];
rxx(0) q[24], q[26];
rx(1.5707963267948966) q[10];
rxx(0) q[6], q[25];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[16];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[28];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[16];
rxx(0) q[28], q[7];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[23];
rxx(0) q[1], q[2];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[11];
rxx(0) q[8], q[17];
rxx(0) q[8], q[25];
rz(1.5707963267948966) q[4];
rxx(0) q[27], q[3];
rz(1.5707963267948966) q[25];
rxx(0) q[13], q[26];
rxx(0) q[1], q[16];
ry(1.5707963267948966) q[13];
rxx(0) q[13], q[2];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[13];
rxx(0) q[23], q[8];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[12];
rxx(0) q[16], q[25];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[26];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[15];
rxx(0) q[17], q[28];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[12];
rxx(0) q[6], q[4];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[23];
rxx(0) q[17], q[2];
rz(1.5707963267948966) q[27];
rx(1.5707963267948966) q[14];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[19];
rxx(0) q[14], q[15];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[11];
rxx(0) q[2], q[21];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[0];
rxx(0) q[0], q[4];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[14];
rxx(0) q[18], q[9];
ry(1.5707963267948966) q[26];
rxx(0) q[6], q[1];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[7];
rxx(0) q[11], q[27];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[12];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[11];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[26];
rxx(0) q[2], q[9];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[13];
rxx(0) q[8], q[13];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[28];
rz(1.5707963267948966) q[12];
rxx(0) q[27], q[4];
ry(1.5707963267948966) q[26];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[23];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[28];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[28];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[28];
rx(1.5707963267948966) q[14];
rxx(0) q[16], q[28];
rx(1.5707963267948966) q[19];
rxx(0) q[5], q[3];
rx(1.5707963267948966) q[27];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[10];
rxx(0) q[0], q[13];
rxx(0) q[2], q[7];
rxx(0) q[20], q[2];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[22];
rx(1.5707963267948966) q[21];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[28];
rz(1.5707963267948966) q[3];
rxx(0) q[11], q[16];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[19];
rz(1.5707963267948966) q[27];
rxx(0) q[24], q[28];
rx(1.5707963267948966) q[2];
rxx(0) q[14], q[19];
ry(1.5707963267948966) q[24];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[26];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[18];
rxx(0) q[16], q[27];
ry(1.5707963267948966) q[18];
rxx(0) q[25], q[2];
rx(1.5707963267948966) q[28];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[23];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[27];
ry(1.5707963267948966) q[9];
rxx(0) q[14], q[28];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[3];
rxx(0) q[5], q[10];
ry(1.5707963267948966) q[26];
rx(1.5707963267948966) q[4];
rxx(0) q[25], q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
rxx(0) q[19], q[23];
rz(1.5707963267948966) q[20];
rxx(0) q[14], q[19];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[28];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[22];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[26];
rz(1.5707963267948966) q[19];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[27];
rxx(0) q[28], q[10];
rx(1.5707963267948966) q[24];
rxx(0) q[16], q[19];
rxx(0) q[3], q[27];
rxx(0) q[14], q[1];
rx(1.5707963267948966) q[28];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[26];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[3];
rxx(0) q[9], q[15];
ry(1.5707963267948966) q[24];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[17];
rxx(0) q[7], q[16];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[27];
rxx(0) q[3], q[28];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[24];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[24];
rxx(0) q[14], q[13];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[22];
rz(1.5707963267948966) q[28];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[24];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[23];
rx(1.5707963267948966) q[15];
rx(1.5707963267948966) q[9];
rxx(0) q[9], q[19];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[16];
rxx(0) q[17], q[28];
ry(1.5707963267948966) q[24];
rxx(0) q[1], q[20];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[20];
rxx(0) q[14], q[21];
rx(1.5707963267948966) q[17];
rxx(0) q[20], q[0];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[23];
rxx(0) q[3], q[6];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[27];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[23];
rxx(0) q[27], q[2];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[26];
rxx(0) q[20], q[13];
rz(1.5707963267948966) q[16];
rx(1.5707963267948966) q[25];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[7];
rxx(0) q[22], q[25];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[25];
rxx(0) q[0], q[7];
ry(1.5707963267948966) q[26];
ry(1.5707963267948966) q[10];
rxx(0) q[24], q[11];
rxx(0) q[21], q[5];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[8];
rxx(0) q[3], q[23];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[7];
rxx(0) q[22], q[14];
rxx(0) q[19], q[27];
ry(1.5707963267948966) q[28];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[14];
rxx(0) q[0], q[16];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[20];
rz(1.5707963267948966) q[20];
ry(1.5707963267948966) q[25];
rxx(0) q[21], q[1];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[18];
rxx(0) q[20], q[12];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[4];
rxx(0) q[4], q[12];
rxx(0) q[9], q[28];
rxx(0) q[5], q[19];
rx(1.5707963267948966) q[3];
rxx(0) q[2], q[18];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[16];
rx(1.5707963267948966) q[10];
rxx(0) q[19], q[7];
ry(1.5707963267948966) q[0];
rxx(0) q[5], q[26];
rxx(0) q[15], q[10];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[28];
rxx(0) q[9], q[17];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[25];
ry(1.5707963267948966) q[19];
rxx(0) q[3], q[7];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[9];
rxx(0) q[1], q[5];
rxx(0) q[1], q[7];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[15];
rxx(0) q[9], q[11];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[0];
rxx(0) q[1], q[6];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[18];
rxx(0) q[3], q[28];
rz(1.5707963267948966) q[26];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[16];
rx(1.5707963267948966) q[20];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[26];
rxx(0) q[10], q[4];
rz(1.5707963267948966) q[22];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[1];
rxx(0) q[8], q[4];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[27];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[17];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
rxx(0) q[5], q[19];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[21];
rxx(0) q[4], q[0];
ry(1.5707963267948966) q[16];
rxx(0) q[1], q[19];
rxx(0) q[6], q[27];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[16];
rxx(0) q[11], q[23];
ry(1.5707963267948966) q[25];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[1];
rxx(0) q[9], q[16];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[22];
rxx(0) q[14], q[13];
rz(1.5707963267948966) q[28];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[28];
rz(1.5707963267948966) q[9];
rxx(0) q[27], q[19];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[13];
rxx(0) q[27], q[18];
ry(1.5707963267948966) q[23];
rxx(0) q[18], q[7];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[2];
rxx(0) q[13], q[9];
ry(1.5707963267948966) q[13];
rxx(0) q[26], q[23];
rz(1.5707963267948966) q[26];
rx(1.5707963267948966) q[26];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[15];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[17];
rxx(0) q[19], q[9];
rz(1.5707963267948966) q[21];
rxx(0) q[24], q[6];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[27];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[21];
rxx(0) q[27], q[4];
rxx(0) q[12], q[24];
ry(1.5707963267948966) q[28];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[12];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[10];
rxx(0) q[9], q[25];
rx(1.5707963267948966) q[0];
rxx(0) q[2], q[14];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[25];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[10];
rxx(0) q[21], q[22];
rxx(0) q[12], q[9];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[26];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[28];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[20];
rxx(0) q[11], q[10];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[21];
rxx(0) q[16], q[13];
rx(1.5707963267948966) q[6];
rxx(0) q[0], q[15];
rxx(0) q[8], q[5];
ry(1.5707963267948966) q[27];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[18];
rxx(0) q[11], q[8];
rz(1.5707963267948966) q[27];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[13];
rxx(0) q[25], q[17];
rz(1.5707963267948966) q[3];
rxx(0) q[17], q[3];
rx(1.5707963267948966) q[26];
rxx(0) q[26], q[19];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[13];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[10];
rxx(0) q[25], q[16];
rxx(0) q[28], q[5];
rx(1.5707963267948966) q[17];
rz(1.5707963267948966) q[25];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[14];
rxx(0) q[13], q[16];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[13];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[11];
rxx(0) q[7], q[8];
ry(1.5707963267948966) q[11];
rxx(0) q[19], q[23];
rxx(0) q[28], q[22];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[19];
rxx(0) q[22], q[11];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[14];
ry(1.5707963267948966) q[5];
rxx(0) q[4], q[10];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[20];
rxx(0) q[2], q[25];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[17];
rxx(0) q[8], q[19];
rx(1.5707963267948966) q[2];
rxx(0) q[26], q[2];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[14];
rx(1.5707963267948966) q[16];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[21];
rz(1.5707963267948966) q[27];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[23];
rxx(0) q[9], q[28];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[15];
rxx(0) q[9], q[11];
rx(1.5707963267948966) q[6];
rxx(0) q[11], q[13];
rxx(0) q[3], q[16];
rxx(0) q[25], q[6];
rz(1.5707963267948966) q[19];
rx(1.5707963267948966) q[28];
ry(1.5707963267948966) q[2];
rxx(0) q[0], q[24];
rxx(0) q[26], q[2];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[17];
rxx(0) q[7], q[24];
rz(1.5707963267948966) q[1];
rxx(0) q[25], q[4];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[0];
rz(1.5707963267948966) q[25];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[26];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[2];
rxx(0) q[10], q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[24];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[27];
rz(1.5707963267948966) q[19];
rxx(0) q[27], q[8];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[10];
rxx(0) q[1], q[28];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[7];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[6];
rxx(0) q[24], q[19];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[26];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[21];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[1];
rxx(0) q[10], q[18];
rz(1.5707963267948966) q[20];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[23];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[21];
rxx(0) q[15], q[6];
