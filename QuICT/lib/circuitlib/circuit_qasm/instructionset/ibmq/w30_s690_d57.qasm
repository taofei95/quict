OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
x q[28];
cx q[9], q[19];
x q[16];
rz(1.5707963267948966) q[20];
x q[3];
x q[26];
sx q[25];
sx q[26];
rz(1.5707963267948966) q[6];
sx q[26];
cx q[9], q[22];
rz(1.5707963267948966) q[29];
rz(1.5707963267948966) q[18];
rz(1.5707963267948966) q[19];
cx q[29], q[26];
x q[8];
sx q[17];
cx q[12], q[25];
cx q[22], q[5];
rz(1.5707963267948966) q[16];
sx q[4];
x q[15];
x q[28];
rz(1.5707963267948966) q[15];
sx q[18];
cx q[6], q[27];
rz(1.5707963267948966) q[6];
cx q[22], q[9];
sx q[2];
x q[5];
x q[26];
cx q[0], q[28];
x q[14];
x q[1];
x q[29];
x q[20];
x q[29];
x q[11];
cx q[6], q[21];
rz(1.5707963267948966) q[29];
rz(1.5707963267948966) q[7];
sx q[6];
cx q[16], q[8];
rz(1.5707963267948966) q[23];
x q[3];
sx q[3];
rz(1.5707963267948966) q[23];
x q[2];
x q[16];
x q[5];
x q[16];
rz(1.5707963267948966) q[26];
sx q[11];
rz(1.5707963267948966) q[16];
x q[2];
x q[0];
sx q[17];
sx q[23];
x q[22];
sx q[0];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[7];
sx q[18];
cx q[18], q[12];
cx q[0], q[26];
x q[24];
sx q[22];
x q[29];
sx q[29];
sx q[20];
x q[23];
rz(1.5707963267948966) q[0];
cx q[27], q[0];
x q[16];
rz(1.5707963267948966) q[6];
cx q[12], q[23];
sx q[13];
rz(1.5707963267948966) q[28];
rz(1.5707963267948966) q[26];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[14];
x q[23];
rz(1.5707963267948966) q[23];
sx q[25];
sx q[2];
rz(1.5707963267948966) q[13];
sx q[21];
x q[22];
sx q[10];
x q[3];
x q[3];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[29];
rz(1.5707963267948966) q[5];
sx q[13];
rz(1.5707963267948966) q[3];
x q[9];
x q[8];
cx q[20], q[22];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[25];
cx q[18], q[19];
sx q[9];
cx q[13], q[20];
x q[2];
cx q[22], q[6];
x q[6];
sx q[26];
sx q[22];
sx q[5];
sx q[6];
sx q[29];
rz(1.5707963267948966) q[16];
sx q[28];
sx q[26];
cx q[3], q[8];
x q[19];
x q[9];
cx q[2], q[15];
x q[12];
x q[7];
x q[27];
cx q[28], q[8];
sx q[19];
x q[2];
x q[2];
sx q[27];
cx q[0], q[16];
rz(1.5707963267948966) q[21];
sx q[5];
sx q[26];
x q[15];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
x q[10];
sx q[3];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[22];
x q[29];
x q[20];
rz(1.5707963267948966) q[25];
x q[9];
sx q[15];
x q[24];
sx q[18];
cx q[4], q[7];
sx q[8];
cx q[1], q[19];
x q[7];
rz(1.5707963267948966) q[10];
cx q[9], q[21];
rz(1.5707963267948966) q[16];
cx q[26], q[15];
sx q[14];
cx q[25], q[2];
x q[18];
sx q[7];
sx q[24];
rz(1.5707963267948966) q[29];
rz(1.5707963267948966) q[19];
sx q[8];
cx q[22], q[0];
x q[16];
cx q[14], q[28];
sx q[24];
x q[29];
x q[28];
sx q[10];
rz(1.5707963267948966) q[8];
cx q[19], q[17];
sx q[27];
sx q[9];
x q[14];
cx q[18], q[8];
rz(1.5707963267948966) q[20];
x q[25];
sx q[1];
sx q[14];
x q[8];
cx q[6], q[4];
rz(1.5707963267948966) q[29];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
cx q[2], q[25];
rz(1.5707963267948966) q[27];
sx q[23];
sx q[10];
sx q[1];
sx q[2];
x q[17];
rz(1.5707963267948966) q[2];
sx q[26];
sx q[2];
sx q[24];
sx q[28];
sx q[29];
sx q[18];
x q[21];
sx q[3];
sx q[5];
sx q[15];
rz(1.5707963267948966) q[12];
x q[15];
sx q[6];
x q[23];
sx q[28];
x q[11];
x q[25];
x q[6];
sx q[23];
x q[16];
rz(1.5707963267948966) q[18];
cx q[8], q[7];
cx q[19], q[28];
rz(1.5707963267948966) q[26];
sx q[10];
rz(1.5707963267948966) q[23];
x q[6];
sx q[1];
sx q[29];
rz(1.5707963267948966) q[12];
sx q[13];
x q[28];
sx q[10];
cx q[24], q[15];
cx q[17], q[11];
cx q[22], q[18];
sx q[28];
x q[19];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[11];
x q[8];
sx q[14];
rz(1.5707963267948966) q[4];
cx q[8], q[27];
rz(1.5707963267948966) q[3];
sx q[6];
sx q[17];
cx q[23], q[19];
sx q[12];
sx q[17];
cx q[29], q[5];
x q[15];
rz(1.5707963267948966) q[13];
sx q[17];
cx q[16], q[27];
x q[3];
cx q[21], q[19];
cx q[2], q[27];
x q[7];
rz(1.5707963267948966) q[28];
sx q[26];
rz(1.5707963267948966) q[18];
cx q[17], q[8];
x q[2];
sx q[9];
x q[17];
sx q[5];
x q[21];
sx q[27];
sx q[25];
x q[16];
cx q[12], q[27];
cx q[27], q[24];
x q[23];
sx q[19];
x q[5];
x q[13];
rz(1.5707963267948966) q[17];
cx q[26], q[29];
cx q[27], q[29];
cx q[16], q[0];
cx q[9], q[5];
x q[17];
rz(1.5707963267948966) q[2];
x q[17];
x q[2];
x q[18];
rz(1.5707963267948966) q[11];
x q[27];
sx q[20];
rz(1.5707963267948966) q[6];
sx q[17];
sx q[0];
cx q[29], q[18];
x q[13];
cx q[13], q[10];
x q[26];
sx q[0];
cx q[18], q[17];
rz(1.5707963267948966) q[3];
cx q[8], q[24];
rz(1.5707963267948966) q[14];
cx q[5], q[8];
sx q[21];
rz(1.5707963267948966) q[28];
rz(1.5707963267948966) q[17];
x q[11];
sx q[11];
sx q[1];
rz(1.5707963267948966) q[2];
sx q[8];
rz(1.5707963267948966) q[10];
sx q[14];
rz(1.5707963267948966) q[24];
cx q[7], q[26];
cx q[22], q[6];
x q[0];
x q[3];
rz(1.5707963267948966) q[29];
rz(1.5707963267948966) q[5];
sx q[25];
x q[13];
cx q[9], q[25];
cx q[14], q[16];
cx q[15], q[26];
sx q[4];
rz(1.5707963267948966) q[13];
cx q[28], q[29];
sx q[14];
x q[1];
rz(1.5707963267948966) q[29];
sx q[16];
sx q[22];
x q[14];
sx q[28];
x q[14];
sx q[13];
sx q[11];
sx q[9];
sx q[10];
x q[17];
sx q[19];
x q[20];
sx q[14];
cx q[29], q[13];
rz(1.5707963267948966) q[14];
x q[11];
rz(1.5707963267948966) q[8];
sx q[21];
rz(1.5707963267948966) q[1];
sx q[6];
x q[8];
x q[8];
sx q[21];
x q[9];
rz(1.5707963267948966) q[23];
rz(1.5707963267948966) q[7];
cx q[16], q[27];
rz(1.5707963267948966) q[6];
sx q[27];
x q[25];
cx q[17], q[27];
rz(1.5707963267948966) q[14];
cx q[26], q[14];
x q[1];
rz(1.5707963267948966) q[18];
x q[20];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[3];
cx q[25], q[20];
cx q[27], q[16];
sx q[8];
x q[1];
x q[8];
cx q[4], q[18];
sx q[5];
x q[14];
rz(1.5707963267948966) q[12];
sx q[15];
cx q[25], q[13];
sx q[0];
cx q[18], q[16];
rz(1.5707963267948966) q[15];
x q[19];
x q[5];
rz(1.5707963267948966) q[20];
cx q[19], q[8];
rz(1.5707963267948966) q[17];
sx q[11];
rz(1.5707963267948966) q[8];
sx q[14];
rz(1.5707963267948966) q[24];
sx q[18];
rz(1.5707963267948966) q[0];
x q[27];
sx q[2];
rz(1.5707963267948966) q[25];
x q[3];
x q[15];
rz(1.5707963267948966) q[18];
sx q[12];
rz(1.5707963267948966) q[1];
sx q[15];
sx q[17];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[18];
sx q[5];
rz(1.5707963267948966) q[11];
sx q[14];
cx q[14], q[24];
cx q[22], q[21];
cx q[9], q[19];
rz(1.5707963267948966) q[2];
sx q[3];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[21];
x q[21];
sx q[23];
x q[16];
cx q[10], q[17];
x q[7];
x q[0];
cx q[21], q[15];
sx q[14];
cx q[19], q[4];
sx q[10];
rz(1.5707963267948966) q[14];
x q[7];
sx q[2];
rz(1.5707963267948966) q[29];
cx q[6], q[10];
sx q[0];
cx q[12], q[3];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[3];
x q[13];
cx q[14], q[13];
x q[21];
sx q[8];
rz(1.5707963267948966) q[19];
cx q[11], q[25];
rz(1.5707963267948966) q[12];
cx q[24], q[8];
rz(1.5707963267948966) q[5];
sx q[5];
sx q[19];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[16];
sx q[19];
cx q[21], q[7];
sx q[20];
rz(1.5707963267948966) q[15];
sx q[16];
cx q[29], q[23];
x q[1];
cx q[6], q[29];
cx q[4], q[22];
x q[28];
sx q[4];
sx q[19];
x q[13];
sx q[9];
cx q[0], q[20];
cx q[25], q[18];
rz(1.5707963267948966) q[21];
cx q[15], q[20];
x q[9];
cx q[28], q[7];
rz(1.5707963267948966) q[18];
sx q[15];
x q[18];
x q[1];
sx q[8];
rz(1.5707963267948966) q[15];
cx q[25], q[12];
x q[27];
cx q[10], q[1];
sx q[6];
x q[1];
rz(1.5707963267948966) q[15];
sx q[1];
rz(1.5707963267948966) q[4];
cx q[0], q[22];
sx q[13];
cx q[17], q[20];
cx q[1], q[8];
sx q[11];
cx q[18], q[10];
rz(1.5707963267948966) q[22];
sx q[3];
sx q[1];
x q[28];
cx q[19], q[1];
rz(1.5707963267948966) q[2];
cx q[10], q[22];
sx q[21];
x q[1];
cx q[1], q[23];
sx q[29];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[16];
cx q[24], q[17];
rz(1.5707963267948966) q[12];
x q[20];
cx q[3], q[23];
cx q[26], q[17];
rz(1.5707963267948966) q[14];
x q[2];
x q[19];
x q[3];
rz(1.5707963267948966) q[20];
x q[5];
cx q[11], q[13];
x q[5];
rz(1.5707963267948966) q[17];
x q[1];
rz(1.5707963267948966) q[10];
sx q[3];
sx q[19];
x q[2];
rz(1.5707963267948966) q[4];
cx q[29], q[5];
x q[7];
x q[24];
x q[4];
x q[21];
cx q[29], q[14];
x q[28];
rz(1.5707963267948966) q[24];
sx q[13];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[4];
x q[8];
x q[1];
rz(1.5707963267948966) q[18];
x q[8];
x q[13];
rz(1.5707963267948966) q[25];
sx q[17];
sx q[26];
x q[25];
sx q[2];
sx q[12];
sx q[13];
sx q[25];
x q[14];
x q[18];
rz(1.5707963267948966) q[2];
sx q[23];
sx q[17];
rz(1.5707963267948966) q[2];
x q[24];
rz(1.5707963267948966) q[21];
x q[10];
cx q[18], q[4];
cx q[12], q[11];
rz(1.5707963267948966) q[9];
cx q[16], q[1];
rz(1.5707963267948966) q[5];
cx q[27], q[3];
rz(1.5707963267948966) q[9];
sx q[4];
sx q[7];
cx q[4], q[11];
sx q[15];
cx q[1], q[29];
sx q[22];
sx q[20];
rz(1.5707963267948966) q[5];
sx q[18];
sx q[19];
rz(1.5707963267948966) q[26];
cx q[25], q[22];
x q[6];
rz(1.5707963267948966) q[16];
x q[2];
rz(1.5707963267948966) q[24];
x q[5];
x q[2];
rz(1.5707963267948966) q[19];
sx q[9];
cx q[6], q[11];
x q[11];
rz(1.5707963267948966) q[10];
sx q[3];
sx q[11];
x q[16];
x q[0];
sx q[1];
x q[18];
cx q[10], q[22];
x q[7];
rz(1.5707963267948966) q[10];
x q[2];
cx q[12], q[27];
x q[1];
x q[25];
cx q[23], q[22];
sx q[10];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[28];
cx q[22], q[6];
rz(1.5707963267948966) q[14];
x q[23];
rz(1.5707963267948966) q[12];
cx q[12], q[4];
cx q[17], q[23];
sx q[9];
sx q[24];
x q[12];
rz(1.5707963267948966) q[1];
sx q[18];
x q[16];
sx q[13];
sx q[23];
rz(1.5707963267948966) q[5];
x q[6];
rz(1.5707963267948966) q[27];
cx q[22], q[20];
sx q[5];
cx q[2], q[16];
sx q[11];
cx q[4], q[14];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[4];
sx q[14];
sx q[21];
sx q[15];
x q[19];
x q[25];
rz(1.5707963267948966) q[15];
cx q[25], q[0];
x q[8];
sx q[0];
cx q[13], q[25];
cx q[10], q[11];
x q[26];
cx q[13], q[22];
x q[7];
rz(1.5707963267948966) q[1];
cx q[1], q[3];
x q[1];
sx q[21];
x q[22];
x q[0];
sx q[26];
sx q[10];
rz(1.5707963267948966) q[24];
sx q[7];
cx q[8], q[2];
rz(1.5707963267948966) q[5];
x q[16];
sx q[27];
sx q[12];
x q[1];
rz(1.5707963267948966) q[29];
sx q[12];
x q[21];
cx q[3], q[12];
x q[24];
sx q[3];
cx q[3], q[7];
cx q[4], q[10];
rz(1.5707963267948966) q[21];
sx q[12];
cx q[0], q[18];
rz(1.5707963267948966) q[7];
sx q[11];
cx q[28], q[23];
sx q[26];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[9];
x q[15];
sx q[7];
x q[20];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[20];
sx q[13];
rz(1.5707963267948966) q[24];
cx q[19], q[21];
sx q[27];
cx q[2], q[9];
cx q[17], q[16];
rz(1.5707963267948966) q[27];
x q[28];
sx q[17];
sx q[13];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[8];
cx q[4], q[21];
x q[16];
rz(1.5707963267948966) q[21];
rz(1.5707963267948966) q[13];
cx q[6], q[27];