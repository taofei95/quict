OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[16];
sx q[15];
cx q[20], q[21];
sx q[2];
rz(1.5707963267948966) q[15];
cx q[12], q[19];
rz(1.5707963267948966) q[11];
x q[9];
cx q[13], q[16];
cx q[10], q[18];
sx q[21];
rz(1.5707963267948966) q[23];
x q[26];
rz(1.5707963267948966) q[5];
cx q[10], q[11];
cx q[16], q[19];
x q[22];
x q[4];
cx q[18], q[2];
sx q[12];
cx q[11], q[21];
x q[5];
rz(1.5707963267948966) q[24];
sx q[23];
rz(1.5707963267948966) q[3];
x q[9];
rz(1.5707963267948966) q[1];
x q[7];
x q[16];
sx q[12];
rz(1.5707963267948966) q[17];
sx q[22];
cx q[23], q[9];
cx q[0], q[9];
rz(1.5707963267948966) q[26];
cx q[24], q[25];
x q[7];
x q[2];
cx q[15], q[10];
sx q[14];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[24];
sx q[19];
rz(1.5707963267948966) q[14];
x q[15];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[25];
cx q[7], q[3];
x q[9];
sx q[11];
rz(1.5707963267948966) q[19];
x q[8];
x q[5];
x q[1];
x q[17];
sx q[19];
cx q[3], q[13];
cx q[11], q[8];
sx q[19];
x q[3];
x q[25];
cx q[20], q[23];
x q[25];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[23];
x q[15];
rz(1.5707963267948966) q[7];
cx q[0], q[24];
rz(1.5707963267948966) q[1];
cx q[23], q[11];
sx q[14];
x q[17];
sx q[26];
sx q[2];
sx q[20];
x q[12];
sx q[17];
x q[1];
cx q[17], q[13];
cx q[4], q[12];
rz(1.5707963267948966) q[9];
sx q[19];
rz(1.5707963267948966) q[26];
sx q[13];
rz(1.5707963267948966) q[17];
sx q[6];
sx q[6];
cx q[16], q[4];
x q[0];
rz(1.5707963267948966) q[23];
x q[14];
rz(1.5707963267948966) q[24];
cx q[15], q[25];
x q[23];
x q[15];
x q[24];
rz(1.5707963267948966) q[3];
x q[3];
rz(1.5707963267948966) q[12];
cx q[22], q[26];
cx q[17], q[21];
cx q[22], q[19];
cx q[12], q[6];
rz(1.5707963267948966) q[21];
sx q[9];
x q[0];
cx q[7], q[14];
sx q[6];
cx q[9], q[19];
x q[14];
sx q[25];
cx q[7], q[5];
x q[26];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[6];
x q[12];
x q[4];
cx q[14], q[21];
rz(1.5707963267948966) q[20];
x q[0];
x q[3];
rz(1.5707963267948966) q[22];
x q[1];
cx q[25], q[16];
x q[2];
x q[15];
x q[24];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[23];
x q[8];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[26];
cx q[24], q[17];
x q[23];
rz(1.5707963267948966) q[12];
sx q[26];
cx q[12], q[9];
sx q[3];
x q[19];
rz(1.5707963267948966) q[6];
sx q[4];
rz(1.5707963267948966) q[22];
x q[11];
sx q[22];
x q[26];
x q[25];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[23];
x q[8];
sx q[2];
x q[7];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[10];
sx q[18];
cx q[12], q[3];
sx q[15];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[20];
sx q[0];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[14];
sx q[15];
cx q[19], q[20];
sx q[19];
sx q[25];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[14];
cx q[6], q[21];
rz(1.5707963267948966) q[16];
x q[2];
cx q[0], q[17];
sx q[16];
cx q[11], q[17];
x q[10];
x q[15];
x q[22];
rz(1.5707963267948966) q[14];
cx q[26], q[8];
cx q[16], q[23];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[20];
x q[17];
sx q[1];
sx q[10];
rz(1.5707963267948966) q[5];
sx q[14];
cx q[7], q[8];
sx q[23];
sx q[22];
cx q[12], q[8];
rz(1.5707963267948966) q[11];
x q[2];
x q[3];
cx q[22], q[18];
sx q[8];
x q[24];
x q[12];
sx q[5];
cx q[20], q[12];
sx q[14];
cx q[24], q[18];
cx q[26], q[3];
rz(1.5707963267948966) q[14];
x q[14];
sx q[6];
sx q[0];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
x q[23];
x q[14];
rz(1.5707963267948966) q[23];
x q[23];
cx q[24], q[16];
cx q[21], q[10];
sx q[15];
rz(1.5707963267948966) q[8];
cx q[20], q[25];
sx q[12];
sx q[4];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[23];
sx q[22];
sx q[24];
sx q[1];
sx q[2];
sx q[3];
cx q[16], q[22];
sx q[26];
sx q[3];
sx q[0];
x q[1];
rz(1.5707963267948966) q[23];
x q[18];
x q[11];
sx q[20];
sx q[21];
sx q[26];
x q[22];
x q[17];
rz(1.5707963267948966) q[21];
x q[1];
cx q[24], q[9];
x q[14];
rz(1.5707963267948966) q[4];
x q[23];
sx q[1];
cx q[1], q[15];
sx q[13];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[1];
x q[25];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[13];
sx q[26];
sx q[6];
x q[7];
cx q[20], q[4];
cx q[16], q[11];
sx q[26];
sx q[8];
sx q[2];
cx q[7], q[15];
cx q[26], q[17];
sx q[16];
rz(1.5707963267948966) q[25];
sx q[0];
sx q[16];
x q[9];
x q[12];
cx q[9], q[25];
sx q[9];
cx q[6], q[18];
sx q[10];
sx q[19];
cx q[20], q[15];
x q[4];
rz(1.5707963267948966) q[5];
x q[11];
sx q[23];
sx q[17];
sx q[14];
cx q[5], q[6];
x q[8];
sx q[16];
rz(1.5707963267948966) q[23];
x q[15];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[14];
sx q[21];
sx q[25];
sx q[9];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[20];
sx q[18];
sx q[1];
x q[24];
rz(1.5707963267948966) q[17];
sx q[8];
cx q[9], q[7];
x q[4];
rz(1.5707963267948966) q[6];
cx q[6], q[3];
x q[9];
x q[3];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[10];
x q[11];
cx q[26], q[11];
sx q[6];
x q[21];
x q[15];
sx q[19];
sx q[12];
sx q[26];
x q[20];
sx q[5];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[17];
sx q[13];
x q[13];
sx q[23];
cx q[17], q[2];
rz(1.5707963267948966) q[15];
sx q[18];
sx q[18];
sx q[20];
x q[11];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
sx q[1];
rz(1.5707963267948966) q[6];
cx q[8], q[15];
rz(1.5707963267948966) q[19];
x q[9];
x q[3];
sx q[7];
sx q[14];
sx q[0];
sx q[9];
x q[26];
cx q[6], q[11];
x q[14];
sx q[4];
x q[26];
x q[11];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[18];
x q[25];
x q[8];
sx q[6];
sx q[3];
cx q[1], q[18];
rz(1.5707963267948966) q[25];
sx q[1];
cx q[4], q[7];
rz(1.5707963267948966) q[11];
cx q[10], q[23];
x q[8];
cx q[14], q[3];
cx q[5], q[3];
rz(1.5707963267948966) q[16];
sx q[21];
sx q[6];
sx q[10];
cx q[8], q[4];
cx q[24], q[17];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[2];
sx q[1];
sx q[24];
cx q[7], q[19];
sx q[10];
sx q[11];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[5];
x q[13];
rz(1.5707963267948966) q[3];
sx q[3];
rz(1.5707963267948966) q[7];
cx q[5], q[18];
x q[14];
x q[11];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[22];
x q[13];
rz(1.5707963267948966) q[15];
sx q[26];
cx q[12], q[4];
rz(1.5707963267948966) q[11];
cx q[10], q[9];
sx q[23];
sx q[2];
rz(1.5707963267948966) q[16];
x q[23];
sx q[24];
rz(1.5707963267948966) q[15];
x q[22];
cx q[0], q[3];
x q[11];
cx q[4], q[7];
x q[19];
x q[20];
sx q[10];
sx q[26];
x q[12];
x q[8];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[21];
sx q[3];
sx q[14];
sx q[16];
sx q[22];
sx q[18];
x q[0];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[21];
sx q[2];
rz(1.5707963267948966) q[14];
sx q[23];
x q[8];
rz(1.5707963267948966) q[7];
sx q[16];
cx q[4], q[1];
x q[18];
rz(1.5707963267948966) q[6];
x q[10];
rz(1.5707963267948966) q[25];
sx q[3];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
cx q[26], q[25];
sx q[5];
sx q[2];
cx q[14], q[19];
sx q[4];
x q[12];
sx q[17];
sx q[13];
cx q[4], q[16];
cx q[17], q[14];
rz(1.5707963267948966) q[15];
sx q[23];
x q[3];
rz(1.5707963267948966) q[13];
cx q[22], q[24];
cx q[12], q[25];
cx q[12], q[16];
x q[3];
sx q[3];
rz(1.5707963267948966) q[15];
sx q[10];
cx q[26], q[19];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
x q[1];
sx q[4];
cx q[15], q[7];
x q[23];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[17];
x q[10];
cx q[23], q[4];
rz(1.5707963267948966) q[25];
sx q[20];
x q[12];
x q[20];
rz(1.5707963267948966) q[4];
cx q[0], q[24];
rz(1.5707963267948966) q[0];
x q[0];
cx q[12], q[20];
cx q[1], q[14];
cx q[26], q[6];
x q[6];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[20];
cx q[3], q[0];
rz(1.5707963267948966) q[6];
sx q[14];
rz(1.5707963267948966) q[18];
x q[10];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[0];
x q[12];
rz(1.5707963267948966) q[20];
sx q[20];
cx q[0], q[5];
x q[4];
cx q[11], q[12];
x q[10];
cx q[7], q[19];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[13];
x q[6];
sx q[4];
sx q[16];
cx q[3], q[25];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[20];
x q[9];
sx q[17];
x q[8];
cx q[1], q[5];
sx q[2];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[12];
sx q[15];
cx q[18], q[3];
x q[5];
sx q[3];
rz(1.5707963267948966) q[11];
sx q[22];
rz(1.5707963267948966) q[10];
cx q[10], q[12];
rz(1.5707963267948966) q[18];
cx q[7], q[26];
x q[19];
cx q[23], q[2];
cx q[15], q[10];
cx q[8], q[18];
cx q[4], q[26];
rz(1.5707963267948966) q[25];
sx q[9];
rz(1.5707963267948966) q[20];
cx q[10], q[16];
cx q[13], q[14];
rz(1.5707963267948966) q[2];
sx q[22];
rz(1.5707963267948966) q[13];
cx q[5], q[4];
rz(1.5707963267948966) q[5];
sx q[9];
cx q[4], q[20];
x q[16];
sx q[1];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[20];
sx q[22];
rz(1.5707963267948966) q[26];
x q[15];
cx q[14], q[3];
rz(1.5707963267948966) q[20];
sx q[19];
x q[16];
rz(1.5707963267948966) q[9];
cx q[16], q[1];
rz(1.5707963267948966) q[10];
x q[9];
cx q[11], q[20];
sx q[3];
sx q[0];
rz(1.5707963267948966) q[6];
cx q[14], q[18];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[0];
x q[1];
cx q[3], q[13];
x q[13];
sx q[23];
rz(1.5707963267948966) q[23];
x q[13];
rz(1.5707963267948966) q[0];
x q[23];
rz(1.5707963267948966) q[5];
sx q[9];
sx q[1];
rz(1.5707963267948966) q[9];
sx q[3];
cx q[1], q[9];
rz(1.5707963267948966) q[21];
cx q[11], q[15];
x q[14];
rz(1.5707963267948966) q[12];
sx q[5];
rz(1.5707963267948966) q[9];
cx q[20], q[0];
x q[12];
sx q[9];
x q[0];
sx q[15];
sx q[11];
cx q[11], q[7];
rz(1.5707963267948966) q[24];
x q[21];
rz(1.5707963267948966) q[24];
x q[24];
x q[1];
x q[23];
rz(1.5707963267948966) q[22];
cx q[6], q[15];
rz(1.5707963267948966) q[2];
sx q[1];
rz(1.5707963267948966) q[17];
x q[10];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[7];
cx q[8], q[13];
x q[11];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[23];
sx q[24];
sx q[9];
cx q[17], q[15];
cx q[21], q[0];
x q[22];
sx q[11];
sx q[10];
cx q[24], q[20];
cx q[3], q[24];
sx q[7];
sx q[14];
x q[17];
x q[24];
cx q[24], q[9];
cx q[19], q[0];
sx q[0];
cx q[4], q[6];
cx q[21], q[15];
cx q[10], q[25];
x q[3];
x q[13];
sx q[8];
rz(1.5707963267948966) q[2];
cx q[9], q[1];
x q[18];
sx q[11];
sx q[6];
x q[14];
sx q[9];
x q[11];
x q[15];
rz(1.5707963267948966) q[24];
rz(1.5707963267948966) q[18];
x q[22];
rz(1.5707963267948966) q[23];
sx q[7];
sx q[25];
sx q[20];
cx q[14], q[0];
sx q[4];
x q[6];
sx q[3];
cx q[6], q[23];
cx q[7], q[21];
cx q[16], q[6];
rz(1.5707963267948966) q[11];
x q[3];
rz(1.5707963267948966) q[9];
cx q[17], q[14];
x q[15];
x q[20];
rz(1.5707963267948966) q[7];
x q[10];
cx q[23], q[10];
x q[25];
sx q[7];
sx q[17];
rz(1.5707963267948966) q[15];
sx q[1];
rz(1.5707963267948966) q[16];
x q[16];
cx q[24], q[5];
cx q[13], q[15];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[19];
sx q[22];
x q[9];
cx q[1], q[8];
cx q[20], q[18];
sx q[9];
x q[25];
x q[1];
x q[21];
x q[15];
cx q[19], q[10];
x q[6];
x q[16];
cx q[22], q[1];
x q[15];
sx q[18];
cx q[17], q[22];
x q[17];
rz(1.5707963267948966) q[0];
x q[22];
rz(1.5707963267948966) q[21];
cx q[26], q[24];
x q[22];
sx q[26];
x q[24];
x q[24];
x q[26];
cx q[11], q[3];
x q[23];
rz(1.5707963267948966) q[16];
sx q[6];
cx q[7], q[22];
sx q[15];
sx q[20];
cx q[26], q[7];
cx q[0], q[4];
sx q[19];
sx q[7];
sx q[3];
sx q[11];
rz(1.5707963267948966) q[1];
x q[2];
cx q[12], q[11];
cx q[4], q[9];
cx q[8], q[2];
sx q[6];
x q[2];
sx q[12];
rz(1.5707963267948966) q[18];
x q[21];
rz(1.5707963267948966) q[6];
sx q[26];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[13];
cx q[7], q[14];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[2];
x q[20];
rz(1.5707963267948966) q[26];
sx q[24];
cx q[0], q[2];
sx q[17];
sx q[14];
x q[14];
x q[5];
sx q[20];
cx q[12], q[9];
cx q[0], q[7];
rz(1.5707963267948966) q[26];
sx q[16];
cx q[24], q[10];
cx q[6], q[20];
x q[15];
sx q[13];
sx q[25];
x q[23];
x q[24];
sx q[17];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
cx q[15], q[20];
cx q[19], q[14];
x q[10];
rz(1.5707963267948966) q[15];
sx q[13];
cx q[22], q[25];
x q[9];
rz(1.5707963267948966) q[12];
cx q[21], q[9];
x q[19];
x q[3];
cx q[7], q[12];
cx q[24], q[21];
cx q[21], q[23];
sx q[16];
x q[6];
x q[13];
x q[10];
sx q[17];
sx q[7];
x q[10];
sx q[0];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[22];
cx q[15], q[21];
cx q[8], q[4];
x q[13];
sx q[8];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[24];
cx q[25], q[21];
rz(1.5707963267948966) q[17];
cx q[19], q[12];
x q[6];
x q[15];
x q[12];
rz(1.5707963267948966) q[20];
sx q[3];
x q[15];
cx q[7], q[25];
sx q[6];
sx q[6];
cx q[21], q[11];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[14];
sx q[8];
sx q[16];
x q[0];
sx q[14];
cx q[25], q[12];
x q[9];
sx q[20];
x q[3];
x q[20];
sx q[5];
rz(1.5707963267948966) q[8];
cx q[11], q[25];
sx q[4];
sx q[11];
sx q[18];
rz(1.5707963267948966) q[8];
cx q[22], q[13];
x q[13];
cx q[4], q[17];
x q[14];
cx q[17], q[24];
x q[13];
x q[9];
x q[16];
sx q[20];
x q[17];
sx q[5];
sx q[12];
sx q[3];
sx q[9];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[7];
sx q[3];
x q[21];
cx q[7], q[26];
x q[25];
cx q[13], q[26];
sx q[17];
cx q[23], q[14];
rz(1.5707963267948966) q[13];
sx q[8];
x q[17];
sx q[2];
sx q[10];
cx q[20], q[11];
x q[5];
sx q[14];
cx q[9], q[10];
x q[15];
x q[12];
sx q[12];
sx q[24];
rz(1.5707963267948966) q[15];
sx q[6];
cx q[1], q[0];
x q[12];
cx q[20], q[15];
rz(1.5707963267948966) q[15];
x q[23];
cx q[4], q[13];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[1];
x q[15];
sx q[15];
sx q[22];
cx q[26], q[17];
sx q[6];
cx q[3], q[25];
cx q[19], q[4];
cx q[17], q[2];
rz(1.5707963267948966) q[11];
sx q[22];
rz(1.5707963267948966) q[12];
x q[19];
cx q[18], q[2];
rz(1.5707963267948966) q[24];
sx q[21];
cx q[18], q[22];
rz(1.5707963267948966) q[12];
x q[9];
sx q[5];
x q[5];
rz(1.5707963267948966) q[25];
sx q[2];
sx q[18];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[20];
x q[21];
x q[23];
x q[8];
x q[19];
sx q[22];
x q[12];
cx q[10], q[20];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[23];
x q[8];
x q[22];
cx q[13], q[10];
x q[0];
sx q[20];
sx q[4];
x q[21];
sx q[7];
rz(1.5707963267948966) q[12];
sx q[18];
rz(1.5707963267948966) q[18];
cx q[2], q[25];
rz(1.5707963267948966) q[16];
x q[9];
x q[15];
rz(1.5707963267948966) q[3];
x q[20];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[19];
x q[7];
x q[6];
sx q[22];
x q[1];
sx q[7];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[18];
cx q[13], q[26];
rz(1.5707963267948966) q[8];
x q[1];
x q[18];
cx q[15], q[22];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[17];
sx q[15];
sx q[4];
x q[14];
rz(1.5707963267948966) q[25];
sx q[11];
cx q[0], q[20];
sx q[7];
rz(1.5707963267948966) q[1];
x q[13];
x q[1];
cx q[11], q[14];
sx q[18];
x q[9];
x q[6];
sx q[3];
sx q[25];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
x q[1];
cx q[2], q[22];
cx q[26], q[10];
sx q[18];
cx q[7], q[0];
x q[3];
x q[10];
sx q[24];
x q[21];
cx q[23], q[25];
cx q[22], q[11];
sx q[26];
cx q[5], q[26];
rz(1.5707963267948966) q[23];
sx q[3];
sx q[7];
sx q[3];
x q[16];
x q[15];
x q[14];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[2];
x q[18];
cx q[4], q[6];
cx q[15], q[5];
x q[24];
rz(1.5707963267948966) q[1];
sx q[23];
cx q[11], q[9];
sx q[7];
cx q[5], q[20];
x q[12];
sx q[19];
sx q[15];
rz(1.5707963267948966) q[17];
sx q[24];
x q[22];
rz(1.5707963267948966) q[2];
sx q[14];
rz(1.5707963267948966) q[11];
sx q[3];
sx q[21];
sx q[9];
sx q[23];
x q[18];
cx q[10], q[14];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[3];
sx q[13];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[1];
cx q[12], q[2];
x q[2];
cx q[11], q[7];
sx q[1];
sx q[3];
x q[5];
x q[12];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[10];
x q[18];
x q[12];
sx q[4];
cx q[8], q[0];
sx q[23];
cx q[15], q[14];
sx q[0];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[25];
sx q[5];
x q[11];
rz(1.5707963267948966) q[22];
x q[6];
rz(1.5707963267948966) q[7];
x q[7];
x q[19];
x q[8];
sx q[26];
cx q[23], q[25];
sx q[15];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[14];
cx q[21], q[8];
cx q[8], q[21];
sx q[19];
cx q[21], q[25];
x q[4];
rz(1.5707963267948966) q[6];
x q[1];
rz(1.5707963267948966) q[5];
cx q[22], q[3];
sx q[26];
x q[2];
x q[18];
cx q[15], q[3];
rz(1.5707963267948966) q[22];
cx q[3], q[0];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[25];
cx q[20], q[13];
cx q[10], q[21];
sx q[16];
cx q[9], q[8];
rz(1.5707963267948966) q[4];
x q[18];
sx q[15];
rz(1.5707963267948966) q[8];
sx q[19];
cx q[22], q[11];
x q[10];
cx q[5], q[14];
x q[25];
cx q[11], q[18];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[3];
sx q[12];
rz(1.5707963267948966) q[8];
sx q[25];
sx q[21];
sx q[2];
cx q[22], q[15];
cx q[18], q[13];
sx q[20];
cx q[16], q[26];
cx q[10], q[23];
rz(1.5707963267948966) q[6];
sx q[3];
x q[4];
x q[19];
x q[24];
x q[26];
sx q[4];
x q[22];
cx q[10], q[5];
cx q[26], q[19];
sx q[14];
sx q[7];
sx q[10];
cx q[25], q[7];
cx q[8], q[18];
rz(1.5707963267948966) q[11];
sx q[9];
sx q[22];
sx q[16];
x q[20];
sx q[19];
cx q[8], q[18];
cx q[15], q[23];
sx q[0];
x q[22];
rz(1.5707963267948966) q[5];
x q[10];
cx q[1], q[9];
sx q[24];
cx q[9], q[12];
x q[19];
cx q[9], q[12];
rz(1.5707963267948966) q[4];
sx q[1];
sx q[26];
rz(1.5707963267948966) q[23];
x q[0];
sx q[12];
sx q[7];
sx q[8];
sx q[18];
sx q[19];
sx q[20];
x q[1];
cx q[4], q[3];
x q[13];
rz(1.5707963267948966) q[18];
sx q[7];
rz(1.5707963267948966) q[16];
x q[10];
sx q[2];
sx q[16];
sx q[8];
sx q[9];
cx q[19], q[26];
sx q[25];
rz(1.5707963267948966) q[24];
x q[22];
cx q[9], q[5];
rz(1.5707963267948966) q[23];
x q[16];
rz(1.5707963267948966) q[22];
cx q[26], q[4];
cx q[1], q[7];
sx q[10];
cx q[24], q[1];
x q[8];
rz(1.5707963267948966) q[25];
x q[12];
sx q[19];
x q[17];
x q[3];
x q[5];
cx q[9], q[4];
rz(1.5707963267948966) q[12];
cx q[17], q[14];
cx q[15], q[0];
sx q[4];
cx q[4], q[5];
sx q[5];
sx q[13];
cx q[22], q[14];
cx q[9], q[4];
sx q[10];
rz(1.5707963267948966) q[17];
cx q[24], q[10];
sx q[7];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[25];
x q[20];
rz(1.5707963267948966) q[23];
x q[17];
cx q[9], q[26];
x q[16];
rz(1.5707963267948966) q[16];
sx q[7];
rz(1.5707963267948966) q[1];
sx q[20];
sx q[8];
cx q[1], q[21];
sx q[19];
sx q[0];
x q[17];
rz(1.5707963267948966) q[15];
cx q[24], q[18];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[15];
cx q[23], q[20];
x q[16];
sx q[15];
sx q[1];
rz(1.5707963267948966) q[5];
x q[16];
sx q[17];
cx q[24], q[1];
cx q[5], q[1];
x q[24];
cx q[1], q[25];
rz(1.5707963267948966) q[6];
x q[14];
cx q[5], q[18];
x q[16];
rz(1.5707963267948966) q[25];
x q[1];
sx q[3];
sx q[6];
x q[9];
rz(1.5707963267948966) q[19];
sx q[14];
cx q[26], q[8];
rz(1.5707963267948966) q[18];
x q[3];
rz(1.5707963267948966) q[8];
x q[15];
rz(1.5707963267948966) q[17];
sx q[17];
sx q[5];
rz(1.5707963267948966) q[18];
x q[4];
sx q[17];
sx q[12];
x q[13];
x q[10];
rz(1.5707963267948966) q[5];
sx q[15];
sx q[6];
cx q[17], q[11];
cx q[1], q[23];
x q[7];
rz(1.5707963267948966) q[4];
cx q[21], q[10];
cx q[8], q[4];
sx q[16];
x q[26];
x q[17];
x q[14];
sx q[24];
rz(1.5707963267948966) q[2];
x q[5];
rz(1.5707963267948966) q[26];
x q[3];
rz(1.5707963267948966) q[22];
x q[9];
x q[3];
x q[6];
cx q[13], q[7];
x q[18];
sx q[10];
cx q[22], q[4];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
sx q[5];
rz(1.5707963267948966) q[19];
cx q[22], q[3];
rz(1.5707963267948966) q[11];
x q[10];
x q[21];
x q[16];
rz(1.5707963267948966) q[0];
cx q[14], q[22];
cx q[26], q[12];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[20];
sx q[26];
rz(1.5707963267948966) q[0];
x q[6];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[17];
sx q[4];
sx q[6];
x q[16];
cx q[16], q[24];
rz(1.5707963267948966) q[9];
cx q[4], q[20];
sx q[5];
cx q[1], q[10];
sx q[24];
cx q[25], q[11];
rz(1.5707963267948966) q[7];
x q[12];
sx q[5];
x q[19];
sx q[3];
cx q[20], q[15];
x q[25];
sx q[10];
cx q[15], q[22];
rz(1.5707963267948966) q[21];
sx q[25];
cx q[3], q[0];
cx q[21], q[2];
cx q[12], q[14];
cx q[20], q[25];
x q[15];
rz(1.5707963267948966) q[4];
sx q[26];
cx q[5], q[13];
sx q[16];
sx q[2];
sx q[10];
rz(1.5707963267948966) q[19];
x q[23];
rz(1.5707963267948966) q[6];
sx q[14];
rz(1.5707963267948966) q[1];
cx q[17], q[22];
sx q[9];
cx q[22], q[21];
rz(1.5707963267948966) q[14];
cx q[19], q[13];
sx q[1];
sx q[25];
sx q[3];
x q[6];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[8];
sx q[18];
x q[12];
x q[13];
cx q[15], q[7];
x q[24];
x q[7];
sx q[14];
cx q[12], q[8];
rz(1.5707963267948966) q[15];
cx q[0], q[18];
sx q[16];
sx q[14];
sx q[7];
rz(1.5707963267948966) q[24];
x q[7];
sx q[8];
x q[12];
sx q[6];
x q[17];
sx q[9];
x q[5];
x q[19];
cx q[6], q[4];
x q[19];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
x q[4];
x q[11];
sx q[22];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[2];
sx q[15];
cx q[4], q[11];
sx q[1];
sx q[5];
sx q[7];
rz(1.5707963267948966) q[1];
sx q[15];
cx q[7], q[14];
x q[3];
sx q[25];
x q[16];
rz(1.5707963267948966) q[10];
sx q[23];
rz(1.5707963267948966) q[3];
sx q[10];
rz(1.5707963267948966) q[25];
x q[5];
cx q[12], q[10];
x q[18];
cx q[15], q[13];
rz(1.5707963267948966) q[16];
sx q[23];
rz(1.5707963267948966) q[12];
x q[22];
x q[7];
cx q[8], q[15];
x q[24];
sx q[15];
x q[4];
sx q[26];
cx q[10], q[8];
x q[16];
rz(1.5707963267948966) q[21];
cx q[24], q[22];
sx q[23];
cx q[19], q[12];
x q[26];
cx q[4], q[21];
cx q[24], q[8];
cx q[16], q[17];
x q[13];
x q[17];
sx q[0];
rz(1.5707963267948966) q[22];
cx q[18], q[6];
x q[23];
x q[15];
sx q[15];
cx q[16], q[13];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[19];
cx q[15], q[13];
cx q[1], q[22];
x q[12];
rz(1.5707963267948966) q[7];
x q[16];
sx q[10];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
x q[12];
x q[20];
x q[16];
sx q[2];
sx q[20];
x q[4];
cx q[9], q[7];
sx q[18];
sx q[0];
rz(1.5707963267948966) q[8];
cx q[16], q[21];
rz(1.5707963267948966) q[2];
x q[20];
x q[1];
cx q[12], q[2];
sx q[5];
cx q[4], q[16];
x q[20];
sx q[7];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[1];
cx q[14], q[26];
rz(1.5707963267948966) q[19];
sx q[21];
cx q[19], q[3];
sx q[22];
x q[12];
cx q[9], q[0];
cx q[25], q[19];
sx q[2];
rz(1.5707963267948966) q[25];
cx q[16], q[25];
rz(1.5707963267948966) q[13];
sx q[14];
cx q[1], q[19];
x q[5];
rz(1.5707963267948966) q[5];
sx q[3];
rz(1.5707963267948966) q[11];
cx q[1], q[26];
x q[21];
sx q[21];
sx q[0];
sx q[20];
rz(1.5707963267948966) q[19];
x q[23];
x q[3];
x q[9];
x q[17];
x q[20];
x q[12];
rz(1.5707963267948966) q[9];
sx q[22];
cx q[4], q[12];
rz(1.5707963267948966) q[24];
sx q[26];
x q[8];
x q[7];
rz(1.5707963267948966) q[20];
x q[26];
x q[8];
sx q[4];
x q[18];
x q[1];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[3];
x q[23];
x q[0];
cx q[8], q[7];
sx q[19];
cx q[7], q[20];
rz(1.5707963267948966) q[23];
sx q[18];
cx q[23], q[22];
cx q[21], q[14];
x q[6];
sx q[5];
x q[0];
cx q[22], q[7];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[10];
sx q[13];
x q[25];
sx q[15];
x q[17];
cx q[4], q[15];
rz(1.5707963267948966) q[11];
x q[3];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[10];
x q[2];
rz(1.5707963267948966) q[12];
sx q[5];
x q[23];
rz(1.5707963267948966) q[11];
x q[24];
sx q[25];
cx q[16], q[0];
cx q[23], q[25];
sx q[10];
x q[7];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[19];
sx q[16];
x q[1];
rz(1.5707963267948966) q[17];
sx q[19];
x q[23];
cx q[26], q[7];
x q[6];
sx q[2];
cx q[6], q[12];
rz(1.5707963267948966) q[3];
sx q[18];
cx q[3], q[10];
sx q[10];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[10];
sx q[5];
x q[18];
sx q[12];
cx q[4], q[16];
sx q[11];
cx q[16], q[21];
rz(1.5707963267948966) q[13];
sx q[4];
cx q[18], q[25];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[21];
x q[8];
cx q[14], q[2];
x q[12];
cx q[20], q[17];
x q[16];
cx q[11], q[10];
x q[13];
sx q[22];
sx q[23];
sx q[21];
x q[12];
sx q[10];
rz(1.5707963267948966) q[20];
sx q[15];
rz(1.5707963267948966) q[4];
cx q[7], q[0];
x q[8];
rz(1.5707963267948966) q[8];
x q[0];
cx q[13], q[24];
x q[22];
x q[17];
cx q[20], q[15];
x q[11];
x q[3];
cx q[20], q[24];
rz(1.5707963267948966) q[3];
cx q[6], q[17];
sx q[23];
rz(1.5707963267948966) q[21];
cx q[10], q[21];
x q[25];
cx q[21], q[5];
sx q[7];
rz(1.5707963267948966) q[3];
x q[9];
sx q[25];
cx q[16], q[2];
cx q[25], q[3];
x q[19];
sx q[2];
rz(1.5707963267948966) q[25];
sx q[2];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[16];
cx q[16], q[7];
cx q[11], q[14];
cx q[12], q[25];
x q[17];
rz(1.5707963267948966) q[17];
cx q[13], q[2];
x q[21];
rz(1.5707963267948966) q[4];
x q[22];
x q[26];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[6];
cx q[21], q[6];
x q[16];
cx q[13], q[23];
rz(1.5707963267948966) q[23];
x q[12];
cx q[2], q[14];
rz(1.5707963267948966) q[7];
x q[9];
sx q[13];
x q[16];
sx q[15];
sx q[17];
sx q[13];
cx q[19], q[12];
x q[9];
sx q[15];
sx q[26];
cx q[25], q[21];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[5];
cx q[13], q[14];
cx q[6], q[14];
sx q[11];
sx q[14];
x q[10];
sx q[7];
x q[5];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[21];
cx q[3], q[25];
cx q[20], q[24];
x q[25];
sx q[1];
cx q[24], q[10];
cx q[13], q[24];
cx q[10], q[24];
sx q[26];
cx q[9], q[26];
x q[18];
cx q[7], q[11];
rz(1.5707963267948966) q[22];
sx q[7];
rz(1.5707963267948966) q[17];
x q[8];
cx q[25], q[6];
x q[5];
x q[25];
cx q[26], q[12];
rz(1.5707963267948966) q[6];
sx q[2];
rz(1.5707963267948966) q[6];
sx q[4];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[12];
cx q[3], q[13];
x q[25];
x q[6];
x q[22];
rz(1.5707963267948966) q[22];
x q[21];
x q[19];
x q[5];
sx q[23];
cx q[20], q[10];
rz(1.5707963267948966) q[10];
x q[17];
sx q[5];
rz(1.5707963267948966) q[25];
x q[17];
sx q[20];
rz(1.5707963267948966) q[18];
cx q[5], q[16];
cx q[25], q[19];
rz(1.5707963267948966) q[9];
cx q[18], q[9];
rz(1.5707963267948966) q[13];
x q[12];
x q[7];
sx q[20];
x q[9];
rz(1.5707963267948966) q[9];
cx q[7], q[9];
rz(1.5707963267948966) q[21];
cx q[10], q[25];
x q[11];
sx q[19];
x q[18];
x q[20];
x q[18];
rz(1.5707963267948966) q[8];
x q[13];
cx q[19], q[12];
rz(1.5707963267948966) q[10];
sx q[22];
rz(1.5707963267948966) q[11];
cx q[1], q[4];
cx q[26], q[22];
x q[1];
rz(1.5707963267948966) q[19];
x q[10];
cx q[17], q[8];
cx q[11], q[21];
cx q[4], q[6];
rz(1.5707963267948966) q[11];
sx q[6];
cx q[9], q[5];
sx q[4];
cx q[5], q[16];
sx q[21];
x q[6];
cx q[9], q[14];
x q[8];
x q[10];
sx q[22];
cx q[22], q[13];
x q[12];
sx q[6];
rz(1.5707963267948966) q[12];
x q[9];
sx q[12];
x q[7];
sx q[19];
cx q[6], q[3];
x q[17];
x q[8];
cx q[11], q[14];
cx q[2], q[6];
sx q[0];
rz(1.5707963267948966) q[25];
sx q[8];
x q[5];
rz(1.5707963267948966) q[18];
cx q[9], q[6];
sx q[15];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
x q[17];
cx q[7], q[22];
rz(1.5707963267948966) q[15];
x q[10];
x q[7];
x q[12];
x q[17];
x q[26];
cx q[10], q[12];
sx q[7];
cx q[11], q[0];
sx q[7];
x q[19];
cx q[4], q[8];
cx q[22], q[17];
sx q[19];
x q[16];
cx q[8], q[21];
rz(1.5707963267948966) q[14];
cx q[16], q[4];
rz(1.5707963267948966) q[19];
x q[25];
cx q[17], q[8];
x q[6];
x q[20];
cx q[3], q[24];
sx q[4];
sx q[20];
rz(1.5707963267948966) q[9];
cx q[22], q[18];
sx q[7];
cx q[2], q[3];
x q[1];
sx q[25];
sx q[12];
rz(1.5707963267948966) q[22];
cx q[15], q[21];
sx q[19];
x q[14];
cx q[11], q[18];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[21];
x q[17];
x q[13];
cx q[0], q[9];
sx q[4];
x q[15];
sx q[16];
x q[26];
x q[12];
rz(1.5707963267948966) q[23];
sx q[3];
sx q[14];
cx q[17], q[15];
sx q[15];
x q[18];
x q[1];
cx q[5], q[12];
cx q[3], q[13];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[8];
sx q[26];
cx q[23], q[14];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[5];
x q[17];
sx q[2];
rz(1.5707963267948966) q[24];
cx q[24], q[0];
sx q[5];
cx q[16], q[7];
sx q[16];
x q[16];
cx q[17], q[12];
cx q[14], q[2];
cx q[20], q[25];
x q[22];
x q[4];
x q[15];
sx q[16];
sx q[0];
rz(1.5707963267948966) q[15];
sx q[7];
rz(1.5707963267948966) q[19];
sx q[24];
x q[20];
cx q[13], q[2];
x q[24];
x q[21];
x q[2];
x q[5];
cx q[17], q[22];
x q[1];
sx q[7];
cx q[23], q[22];
sx q[6];
sx q[21];
cx q[17], q[8];
cx q[9], q[7];
sx q[26];
x q[1];
x q[9];
cx q[10], q[13];
rz(1.5707963267948966) q[3];
cx q[12], q[6];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[25];
x q[13];
x q[24];
rz(1.5707963267948966) q[13];
cx q[6], q[17];
sx q[7];
rz(1.5707963267948966) q[9];
x q[8];
rz(1.5707963267948966) q[13];
x q[13];
x q[24];
x q[1];
sx q[1];
sx q[5];
x q[24];
cx q[23], q[6];
cx q[5], q[12];
cx q[9], q[20];
cx q[24], q[11];
x q[25];
sx q[19];
sx q[12];
sx q[4];
sx q[6];
rz(1.5707963267948966) q[7];
cx q[10], q[13];
cx q[21], q[26];
rz(1.5707963267948966) q[4];
cx q[14], q[16];
sx q[5];
sx q[16];
rz(1.5707963267948966) q[5];
x q[2];
x q[25];
sx q[1];
cx q[20], q[1];
x q[18];
sx q[3];
rz(1.5707963267948966) q[12];
sx q[20];
rz(1.5707963267948966) q[21];
x q[10];
sx q[9];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[19];
cx q[26], q[13];
x q[22];
rz(1.5707963267948966) q[16];
cx q[0], q[16];
x q[25];
rz(1.5707963267948966) q[16];
cx q[20], q[12];
sx q[15];
x q[0];
x q[17];
rz(1.5707963267948966) q[23];
x q[9];
x q[21];
x q[12];
rz(1.5707963267948966) q[11];
sx q[15];
sx q[19];
sx q[20];
sx q[11];
x q[2];
sx q[9];
rz(1.5707963267948966) q[7];
x q[23];
x q[5];
cx q[5], q[12];
cx q[22], q[26];
x q[14];
sx q[0];
x q[5];
cx q[11], q[4];
sx q[2];
cx q[23], q[11];
x q[20];
cx q[9], q[22];
cx q[18], q[15];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[14];
x q[17];
sx q[26];
rz(1.5707963267948966) q[22];
cx q[3], q[9];
sx q[14];
sx q[19];
x q[0];
x q[0];
sx q[25];
rz(1.5707963267948966) q[26];
cx q[13], q[16];
x q[16];
sx q[9];
cx q[8], q[19];
sx q[11];
rz(1.5707963267948966) q[19];
x q[2];
sx q[18];
cx q[17], q[13];
sx q[2];
rz(1.5707963267948966) q[15];
x q[15];
sx q[10];
x q[12];
rz(1.5707963267948966) q[25];
x q[21];
x q[25];
x q[14];
sx q[23];
x q[3];
cx q[9], q[1];
sx q[4];
sx q[6];
cx q[14], q[17];
cx q[9], q[2];
cx q[6], q[14];
x q[17];
x q[8];
x q[17];
cx q[26], q[10];
x q[4];
x q[4];
rz(1.5707963267948966) q[3];
cx q[0], q[5];
x q[1];
rz(1.5707963267948966) q[26];
x q[22];
rz(1.5707963267948966) q[23];
cx q[13], q[11];
sx q[26];
rz(1.5707963267948966) q[20];
cx q[1], q[13];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[17];
sx q[8];
cx q[21], q[19];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[23];
cx q[24], q[9];
sx q[7];
sx q[15];
sx q[11];
rz(1.5707963267948966) q[8];
sx q[3];
x q[4];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[23];
x q[7];
x q[5];
sx q[15];
sx q[17];
x q[23];
cx q[14], q[26];
rz(1.5707963267948966) q[19];
cx q[12], q[23];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[6];
sx q[4];
rz(1.5707963267948966) q[6];
cx q[24], q[26];
cx q[16], q[8];
sx q[1];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[21];
cx q[25], q[8];
cx q[7], q[1];
cx q[20], q[17];
x q[26];
x q[22];
sx q[16];
cx q[2], q[5];
sx q[22];
x q[22];
x q[21];
cx q[10], q[21];
x q[8];
rz(1.5707963267948966) q[13];
x q[3];
rz(1.5707963267948966) q[19];
sx q[0];
cx q[20], q[3];
x q[16];
sx q[24];
cx q[7], q[16];
sx q[6];
cx q[9], q[13];
sx q[3];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[12];
x q[6];
sx q[17];
sx q[15];
sx q[3];
rz(1.5707963267948966) q[26];
cx q[25], q[18];
sx q[26];
cx q[16], q[14];
cx q[2], q[25];
sx q[2];
rz(1.5707963267948966) q[10];
x q[13];
sx q[17];
sx q[3];
x q[23];
x q[26];
cx q[8], q[12];
sx q[3];
sx q[19];
rz(1.5707963267948966) q[4];
sx q[1];
cx q[22], q[10];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[14];
cx q[16], q[26];
sx q[20];
x q[24];
sx q[26];
rz(1.5707963267948966) q[2];
cx q[11], q[23];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[15];
cx q[3], q[13];
rz(1.5707963267948966) q[7];
cx q[0], q[15];
cx q[16], q[0];
cx q[9], q[16];
x q[23];
rz(1.5707963267948966) q[6];
cx q[6], q[19];
x q[21];
cx q[21], q[5];
sx q[5];
cx q[17], q[15];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[25];
sx q[22];
cx q[7], q[10];
sx q[4];
sx q[21];
x q[26];
sx q[6];
sx q[17];
sx q[7];
x q[19];
rz(1.5707963267948966) q[26];
sx q[22];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[2];
sx q[23];
cx q[20], q[9];
cx q[13], q[9];
cx q[1], q[18];
x q[24];
x q[3];
sx q[20];
rz(1.5707963267948966) q[9];
cx q[10], q[15];
sx q[4];
sx q[23];
rz(1.5707963267948966) q[6];
cx q[6], q[3];
cx q[24], q[22];
cx q[3], q[0];
rz(1.5707963267948966) q[19];
x q[24];
sx q[9];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[17];
sx q[12];
x q[18];
rz(1.5707963267948966) q[23];
x q[2];
cx q[5], q[0];
cx q[6], q[2];
sx q[2];
cx q[5], q[6];
x q[9];
sx q[1];
sx q[16];
cx q[2], q[6];
x q[25];
x q[19];
x q[18];
sx q[19];
sx q[17];
x q[6];
rz(1.5707963267948966) q[6];
x q[24];
cx q[13], q[1];
cx q[23], q[18];
cx q[12], q[1];
sx q[25];
rz(1.5707963267948966) q[23];
x q[19];
