OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
h q[14];
h q[15];
h q[16];
h q[17];
h q[18];
h q[19];
h q[20];
h q[21];
h q[22];
h q[23];
h q[24];
h q[25];
h q[26];
h q[27];
h q[28];
h q[29];
cx q[16], q[11];
rz(-1.1226317882537842) q[11];
cx q[16], q[11];
cx q[19], q[16];
rz(-1.1226317882537842) q[16];
cx q[19], q[16];
cx q[16], q[28];
rz(-1.1226317882537842) q[28];
cx q[16], q[28];
cx q[5], q[4];
rz(-1.1226317882537842) q[4];
cx q[5], q[4];
cx q[20], q[16];
rz(-1.1226317882537842) q[16];
cx q[20], q[16];
cx q[0], q[4];
rz(-1.1226317882537842) q[4];
cx q[0], q[4];
cx q[28], q[8];
rz(-1.1226317882537842) q[8];
cx q[28], q[8];
cx q[9], q[18];
rz(-1.1226317882537842) q[18];
cx q[9], q[18];
cx q[10], q[20];
rz(-1.1226317882537842) q[20];
cx q[10], q[20];
cx q[28], q[26];
rz(-1.1226317882537842) q[26];
cx q[28], q[26];
cx q[12], q[17];
rz(-1.1226317882537842) q[17];
cx q[12], q[17];
cx q[2], q[28];
rz(-1.1226317882537842) q[28];
cx q[2], q[28];
cx q[5], q[2];
rz(-1.1226317882537842) q[2];
cx q[5], q[2];
cx q[19], q[7];
rz(-1.1226317882537842) q[7];
cx q[19], q[7];
cx q[1], q[15];
rz(-1.1226317882537842) q[15];
cx q[1], q[15];
cx q[24], q[2];
rz(-1.1226317882537842) q[2];
cx q[24], q[2];
cx q[19], q[22];
rz(-1.1226317882537842) q[22];
cx q[19], q[22];
cx q[19], q[5];
rz(-1.1226317882537842) q[5];
cx q[19], q[5];
cx q[14], q[23];
rz(-1.1226317882537842) q[23];
cx q[14], q[23];
cx q[13], q[29];
rz(-1.1226317882537842) q[29];
cx q[13], q[29];
cx q[0], q[25];
rz(-1.1226317882537842) q[25];
cx q[0], q[25];
cx q[1], q[16];
rz(-1.1226317882537842) q[16];
cx q[1], q[16];
cx q[7], q[23];
rz(-1.1226317882537842) q[23];
cx q[7], q[23];
cx q[26], q[15];
rz(-1.1226317882537842) q[15];
cx q[26], q[15];
cx q[17], q[25];
rz(-1.1226317882537842) q[25];
cx q[17], q[25];
cx q[17], q[26];
rz(-1.1226317882537842) q[26];
cx q[17], q[26];
cx q[20], q[2];
rz(-1.1226317882537842) q[2];
cx q[20], q[2];
cx q[15], q[19];
rz(-1.1226317882537842) q[19];
cx q[15], q[19];
cx q[15], q[25];
rz(-1.1226317882537842) q[25];
cx q[15], q[25];
cx q[9], q[10];
rz(-1.1226317882537842) q[10];
cx q[9], q[10];
cx q[17], q[15];
rz(-1.1226317882537842) q[15];
cx q[17], q[15];
cx q[13], q[12];
rz(-1.1226317882537842) q[12];
cx q[13], q[12];
cx q[3], q[15];
rz(-1.1226317882537842) q[15];
cx q[3], q[15];
cx q[24], q[21];
rz(-1.1226317882537842) q[21];
cx q[24], q[21];
cx q[7], q[28];
rz(-1.1226317882537842) q[28];
cx q[7], q[28];
cx q[8], q[26];
rz(-1.1226317882537842) q[26];
cx q[8], q[26];
cx q[25], q[20];
rz(-1.1226317882537842) q[20];
cx q[25], q[20];
cx q[22], q[17];
rz(-1.1226317882537842) q[17];
cx q[22], q[17];
cx q[14], q[24];
rz(-1.1226317882537842) q[24];
cx q[14], q[24];
cx q[20], q[0];
rz(-1.1226317882537842) q[0];
cx q[20], q[0];
cx q[29], q[14];
rz(-1.1226317882537842) q[14];
cx q[29], q[14];
cx q[18], q[15];
rz(-1.1226317882537842) q[15];
cx q[18], q[15];
cx q[18], q[8];
rz(-1.1226317882537842) q[8];
cx q[18], q[8];
cx q[1], q[6];
rz(-1.1226317882537842) q[6];
cx q[1], q[6];
cx q[5], q[12];
rz(-1.1226317882537842) q[12];
cx q[5], q[12];
cx q[7], q[8];
rz(-1.1226317882537842) q[8];
cx q[7], q[8];
cx q[6], q[26];
rz(-1.1226317882537842) q[26];
cx q[6], q[26];
cx q[19], q[1];
rz(-1.1226317882537842) q[1];
cx q[19], q[1];
cx q[13], q[21];
rz(-1.1226317882537842) q[21];
cx q[13], q[21];
cx q[29], q[11];
rz(-1.1226317882537842) q[11];
cx q[29], q[11];
cx q[24], q[19];
rz(-1.1226317882537842) q[19];
cx q[24], q[19];
cx q[29], q[7];
rz(-1.1226317882537842) q[7];
cx q[29], q[7];
cx q[7], q[1];
rz(-1.1226317882537842) q[1];
cx q[7], q[1];
cx q[1], q[0];
rz(-1.1226317882537842) q[0];
cx q[1], q[0];
cx q[4], q[3];
rz(-1.1226317882537842) q[3];
cx q[4], q[3];
cx q[0], q[2];
rz(-1.1226317882537842) q[2];
cx q[0], q[2];
cx q[11], q[18];
rz(-1.1226317882537842) q[18];
cx q[11], q[18];
cx q[29], q[20];
rz(-1.1226317882537842) q[20];
cx q[29], q[20];
cx q[6], q[15];
rz(-1.1226317882537842) q[15];
cx q[6], q[15];
cx q[9], q[6];
rz(-1.1226317882537842) q[6];
cx q[9], q[6];
cx q[9], q[27];
rz(-1.1226317882537842) q[27];
cx q[9], q[27];
cx q[14], q[12];
rz(-1.1226317882537842) q[12];
cx q[14], q[12];
cx q[17], q[13];
rz(-1.1226317882537842) q[13];
cx q[17], q[13];
cx q[6], q[3];
rz(-1.1226317882537842) q[3];
cx q[6], q[3];
cx q[24], q[28];
rz(-1.1226317882537842) q[28];
cx q[24], q[28];
cx q[5], q[21];
rz(-1.1226317882537842) q[21];
cx q[5], q[21];
cx q[19], q[29];
rz(-1.1226317882537842) q[29];
cx q[19], q[29];
cx q[18], q[17];
rz(-1.1226317882537842) q[17];
cx q[18], q[17];
cx q[23], q[11];
rz(-1.1226317882537842) q[11];
cx q[23], q[11];
cx q[10], q[23];
rz(-1.1226317882537842) q[23];
cx q[10], q[23];
cx q[8], q[22];
rz(-1.1226317882537842) q[22];
cx q[8], q[22];
cx q[9], q[17];
rz(-1.1226317882537842) q[17];
cx q[9], q[17];
cx q[16], q[25];
rz(-1.1226317882537842) q[25];
cx q[16], q[25];
cx q[13], q[15];
rz(-1.1226317882537842) q[15];
cx q[13], q[15];
cx q[22], q[1];
rz(-1.1226317882537842) q[1];
cx q[22], q[1];
cx q[15], q[28];
rz(-1.1226317882537842) q[28];
cx q[15], q[28];
cx q[16], q[18];
rz(-1.1226317882537842) q[18];
cx q[16], q[18];
cx q[28], q[10];
rz(-1.1226317882537842) q[10];
cx q[28], q[10];
cx q[0], q[22];
rz(-1.1226317882537842) q[22];
cx q[0], q[22];
cx q[23], q[26];
rz(-1.1226317882537842) q[26];
cx q[23], q[26];
cx q[29], q[10];
rz(-1.1226317882537842) q[10];
cx q[29], q[10];
cx q[5], q[9];
rz(-1.1226317882537842) q[9];
cx q[5], q[9];
cx q[1], q[5];
rz(-1.1226317882537842) q[5];
cx q[1], q[5];
cx q[20], q[7];
rz(-1.1226317882537842) q[7];
cx q[20], q[7];
cx q[7], q[9];
rz(-1.1226317882537842) q[9];
cx q[7], q[9];
cx q[27], q[19];
rz(-1.1226317882537842) q[19];
cx q[27], q[19];
cx q[6], q[29];
rz(-1.1226317882537842) q[29];
cx q[6], q[29];
cx q[0], q[21];
rz(-1.1226317882537842) q[21];
cx q[0], q[21];
cx q[23], q[15];
rz(-1.1226317882537842) q[15];
cx q[23], q[15];
cx q[8], q[14];
rz(-1.1226317882537842) q[14];
cx q[8], q[14];
cx q[13], q[0];
rz(-1.1226317882537842) q[0];
cx q[13], q[0];
cx q[24], q[11];
rz(-1.1226317882537842) q[11];
cx q[24], q[11];
cx q[9], q[28];
rz(-1.1226317882537842) q[28];
cx q[9], q[28];
cx q[4], q[15];
rz(-1.1226317882537842) q[15];
cx q[4], q[15];
cx q[3], q[22];
rz(-1.1226317882537842) q[22];
cx q[3], q[22];
cx q[2], q[29];
rz(-1.1226317882537842) q[29];
cx q[2], q[29];
cx q[11], q[14];
rz(-1.1226317882537842) q[14];
cx q[11], q[14];
cx q[3], q[14];
rz(-1.1226317882537842) q[14];
cx q[3], q[14];
cx q[12], q[3];
rz(-1.1226317882537842) q[3];
cx q[12], q[3];
cx q[29], q[27];
rz(-1.1226317882537842) q[27];
cx q[29], q[27];
cx q[6], q[20];
rz(-1.1226317882537842) q[20];
cx q[6], q[20];
cx q[10], q[22];
rz(-1.1226317882537842) q[22];
cx q[10], q[22];
cx q[14], q[18];
rz(-1.1226317882537842) q[18];
cx q[14], q[18];
cx q[3], q[2];
rz(-1.1226317882537842) q[2];
cx q[3], q[2];
cx q[24], q[26];
rz(-1.1226317882537842) q[26];
cx q[24], q[26];
cx q[16], q[0];
rz(-1.1226317882537842) q[0];
cx q[16], q[0];
cx q[4], q[1];
rz(-1.1226317882537842) q[1];
cx q[4], q[1];
cx q[23], q[20];
rz(-1.1226317882537842) q[20];
cx q[23], q[20];
cx q[19], q[18];
rz(-1.1226317882537842) q[18];
cx q[19], q[18];
cx q[0], q[3];
rz(-1.1226317882537842) q[3];
cx q[0], q[3];
cx q[11], q[28];
rz(-1.1226317882537842) q[28];
cx q[11], q[28];
cx q[23], q[24];
rz(-1.1226317882537842) q[24];
cx q[23], q[24];
cx q[1], q[8];
rz(-1.1226317882537842) q[8];
cx q[1], q[8];
cx q[21], q[11];
rz(-1.1226317882537842) q[11];
cx q[21], q[11];
cx q[14], q[26];
rz(-1.1226317882537842) q[26];
cx q[14], q[26];
cx q[28], q[13];
rz(-1.1226317882537842) q[13];
cx q[28], q[13];
cx q[15], q[29];
rz(-1.1226317882537842) q[29];
cx q[15], q[29];
cx q[28], q[19];
rz(-1.1226317882537842) q[19];
cx q[28], q[19];
cx q[26], q[11];
rz(-1.1226317882537842) q[11];
cx q[26], q[11];
cx q[13], q[25];
rz(-1.1226317882537842) q[25];
cx q[13], q[25];
cx q[11], q[15];
rz(-1.1226317882537842) q[15];
cx q[11], q[15];
cx q[17], q[19];
rz(-1.1226317882537842) q[19];
cx q[17], q[19];
cx q[4], q[10];
rz(-1.1226317882537842) q[10];
cx q[4], q[10];
cx q[16], q[21];
rz(-1.1226317882537842) q[21];
cx q[16], q[21];
cx q[6], q[16];
rz(-1.1226317882537842) q[16];
cx q[6], q[16];
cx q[2], q[14];
rz(-1.1226317882537842) q[14];
cx q[2], q[14];
cx q[20], q[12];
rz(-1.1226317882537842) q[12];
cx q[20], q[12];
cx q[21], q[26];
rz(-1.1226317882537842) q[26];
cx q[21], q[26];
cx q[27], q[13];
rz(-1.1226317882537842) q[13];
cx q[27], q[13];
cx q[5], q[16];
rz(-1.1226317882537842) q[16];
cx q[5], q[16];
cx q[1], q[3];
rz(-1.1226317882537842) q[3];
cx q[1], q[3];
cx q[11], q[13];
rz(-1.1226317882537842) q[13];
cx q[11], q[13];
cx q[1], q[12];
rz(-1.1226317882537842) q[12];
cx q[1], q[12];
cx q[16], q[7];
rz(-1.1226317882537842) q[7];
cx q[16], q[7];
cx q[3], q[21];
rz(-1.1226317882537842) q[21];
cx q[3], q[21];
cx q[0], q[5];
rz(-1.1226317882537842) q[5];
cx q[0], q[5];
cx q[1], q[23];
rz(-1.1226317882537842) q[23];
cx q[1], q[23];
cx q[13], q[8];
rz(-1.1226317882537842) q[8];
cx q[13], q[8];
cx q[4], q[8];
rz(-1.1226317882537842) q[8];
cx q[4], q[8];
cx q[4], q[29];
rz(-1.1226317882537842) q[29];
cx q[4], q[29];
cx q[12], q[4];
rz(-1.1226317882537842) q[4];
cx q[12], q[4];
cx q[28], q[5];
rz(-1.1226317882537842) q[5];
cx q[28], q[5];
cx q[28], q[25];
rz(-1.1226317882537842) q[25];
cx q[28], q[25];
cx q[18], q[29];
rz(-1.1226317882537842) q[29];
cx q[18], q[29];
cx q[4], q[26];
rz(-1.1226317882537842) q[26];
cx q[4], q[26];
cx q[0], q[14];
rz(-1.1226317882537842) q[14];
cx q[0], q[14];
cx q[24], q[3];
rz(-1.1226317882537842) q[3];
cx q[24], q[3];
cx q[6], q[22];
rz(-1.1226317882537842) q[22];
cx q[6], q[22];
cx q[17], q[3];
rz(-1.1226317882537842) q[3];
cx q[17], q[3];
cx q[6], q[19];
rz(-1.1226317882537842) q[19];
cx q[6], q[19];
cx q[1], q[20];
rz(-1.1226317882537842) q[20];
cx q[1], q[20];
cx q[8], q[25];
rz(-1.1226317882537842) q[25];
cx q[8], q[25];
cx q[10], q[17];
rz(-1.1226317882537842) q[17];
cx q[10], q[17];
cx q[7], q[2];
rz(-1.1226317882537842) q[2];
cx q[7], q[2];
cx q[11], q[22];
rz(-1.1226317882537842) q[22];
cx q[11], q[22];
cx q[19], q[21];
rz(-1.1226317882537842) q[21];
cx q[19], q[21];
cx q[24], q[9];
rz(-1.1226317882537842) q[9];
cx q[24], q[9];
cx q[11], q[25];
rz(-1.1226317882537842) q[25];
cx q[11], q[25];
cx q[28], q[27];
rz(-1.1226317882537842) q[27];
cx q[28], q[27];
cx q[11], q[27];
rz(-1.1226317882537842) q[27];
cx q[11], q[27];
cx q[22], q[25];
rz(-1.1226317882537842) q[25];
cx q[22], q[25];
cx q[12], q[10];
rz(-1.1226317882537842) q[10];
cx q[12], q[10];
cx q[1], q[13];
rz(-1.1226317882537842) q[13];
cx q[1], q[13];
cx q[1], q[10];
rz(-1.1226317882537842) q[10];
cx q[1], q[10];
cx q[10], q[14];
rz(-1.1226317882537842) q[14];
cx q[10], q[14];
cx q[29], q[25];
rz(-1.1226317882537842) q[25];
cx q[29], q[25];
cx q[1], q[24];
rz(-1.1226317882537842) q[24];
cx q[1], q[24];
cx q[21], q[8];
rz(-1.1226317882537842) q[8];
cx q[21], q[8];
cx q[8], q[15];
rz(-1.1226317882537842) q[15];
cx q[8], q[15];
cx q[11], q[7];
rz(-1.1226317882537842) q[7];
cx q[11], q[7];
cx q[9], q[2];
rz(-1.1226317882537842) q[2];
cx q[9], q[2];
cx q[26], q[27];
rz(-1.1226317882537842) q[27];
cx q[26], q[27];
cx q[16], q[12];
rz(-1.1226317882537842) q[12];
cx q[16], q[12];
cx q[9], q[15];
rz(-1.1226317882537842) q[15];
cx q[9], q[15];
cx q[23], q[19];
rz(-1.1226317882537842) q[19];
cx q[23], q[19];
cx q[24], q[22];
rz(-1.1226317882537842) q[22];
cx q[24], q[22];
cx q[4], q[24];
rz(-1.1226317882537842) q[24];
cx q[4], q[24];
cx q[29], q[0];
rz(-1.1226317882537842) q[0];
cx q[29], q[0];
cx q[3], q[16];
rz(-1.1226317882537842) q[16];
cx q[3], q[16];
cx q[12], q[29];
rz(-1.1226317882537842) q[29];
cx q[12], q[29];
cx q[0], q[10];
rz(-1.1226317882537842) q[10];
cx q[0], q[10];
cx q[18], q[28];
rz(-1.1226317882537842) q[28];
cx q[18], q[28];
cx q[29], q[21];
rz(-1.1226317882537842) q[21];
cx q[29], q[21];
cx q[13], q[18];
rz(-1.1226317882537842) q[18];
cx q[13], q[18];
cx q[23], q[28];
rz(-1.1226317882537842) q[28];
cx q[23], q[28];
cx q[6], q[13];
rz(-1.1226317882537842) q[13];
cx q[6], q[13];
cx q[27], q[18];
rz(-1.1226317882537842) q[18];
cx q[27], q[18];
cx q[15], q[22];
rz(-1.1226317882537842) q[22];
cx q[15], q[22];
cx q[9], q[0];
rz(-1.1226317882537842) q[0];
cx q[9], q[0];
cx q[25], q[26];
rz(-1.1226317882537842) q[26];
cx q[25], q[26];
cx q[9], q[19];
rz(-1.1226317882537842) q[19];
cx q[9], q[19];
cx q[0], q[7];
rz(-1.1226317882537842) q[7];
cx q[0], q[7];
cx q[28], q[17];
rz(-1.1226317882537842) q[17];
cx q[28], q[17];
cx q[14], q[7];
rz(-1.1226317882537842) q[7];
cx q[14], q[7];
cx q[4], q[6];
rz(-1.1226317882537842) q[6];
cx q[4], q[6];
cx q[5], q[14];
rz(-1.1226317882537842) q[14];
cx q[5], q[14];
cx q[19], q[25];
rz(-1.1226317882537842) q[25];
cx q[19], q[25];
cx q[11], q[1];
rz(-1.1226317882537842) q[1];
cx q[11], q[1];
cx q[29], q[17];
rz(-1.1226317882537842) q[17];
cx q[29], q[17];
cx q[17], q[16];
rz(-1.1226317882537842) q[16];
cx q[17], q[16];
cx q[20], q[3];
rz(-1.1226317882537842) q[3];
cx q[20], q[3];
cx q[10], q[21];
rz(-1.1226317882537842) q[21];
cx q[10], q[21];
cx q[22], q[4];
rz(-1.1226317882537842) q[4];
cx q[22], q[4];
cx q[1], q[25];
rz(-1.1226317882537842) q[25];
cx q[1], q[25];
cx q[9], q[11];
rz(-1.1226317882537842) q[11];
cx q[9], q[11];
cx q[8], q[19];
rz(-1.1226317882537842) q[19];
cx q[8], q[19];
cx q[18], q[7];
rz(-1.1226317882537842) q[7];
cx q[18], q[7];
cx q[14], q[13];
rz(-1.1226317882537842) q[13];
cx q[14], q[13];
cx q[26], q[9];
rz(-1.1226317882537842) q[9];
cx q[26], q[9];
cx q[26], q[18];
rz(-1.1226317882537842) q[18];
cx q[26], q[18];
cx q[27], q[16];
rz(-1.1226317882537842) q[16];
cx q[27], q[16];
cx q[27], q[2];
rz(-1.1226317882537842) q[2];
cx q[27], q[2];
cx q[20], q[18];
rz(-1.1226317882537842) q[18];
cx q[20], q[18];
cx q[27], q[5];
rz(-1.1226317882537842) q[5];
cx q[27], q[5];
cx q[24], q[10];
rz(-1.1226317882537842) q[10];
cx q[24], q[10];
cx q[28], q[20];
rz(-1.1226317882537842) q[20];
cx q[28], q[20];
cx q[26], q[3];
rz(-1.1226317882537842) q[3];
cx q[26], q[3];
rx(0.955446720123291) q[0];
rx(0.955446720123291) q[1];
rx(0.955446720123291) q[2];
rx(0.955446720123291) q[3];
rx(0.955446720123291) q[4];
rx(0.955446720123291) q[5];
rx(0.955446720123291) q[6];
rx(0.955446720123291) q[7];
rx(0.955446720123291) q[8];
rx(0.955446720123291) q[9];
rx(0.955446720123291) q[10];
rx(0.955446720123291) q[11];
rx(0.955446720123291) q[12];
rx(0.955446720123291) q[13];
rx(0.955446720123291) q[14];
rx(0.955446720123291) q[15];
rx(0.955446720123291) q[16];
rx(0.955446720123291) q[17];
rx(0.955446720123291) q[18];
rx(0.955446720123291) q[19];
rx(0.955446720123291) q[20];
rx(0.955446720123291) q[21];
rx(0.955446720123291) q[22];
rx(0.955446720123291) q[23];
rx(0.955446720123291) q[24];
rx(0.955446720123291) q[25];
rx(0.955446720123291) q[26];
rx(0.955446720123291) q[27];
rx(0.955446720123291) q[28];
rx(0.955446720123291) q[29];