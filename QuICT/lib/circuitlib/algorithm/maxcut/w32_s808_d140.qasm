OPENQASM 2.0;
include "qelib1.inc";
qreg q[32];
creg c[32];
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
h q[30];
h q[31];
cx q[26], q[10];
rz(-0.7197777032852173) q[10];
cx q[26], q[10];
cx q[7], q[26];
rz(-0.7197777032852173) q[26];
cx q[7], q[26];
cx q[5], q[8];
rz(-0.7197777032852173) q[8];
cx q[5], q[8];
cx q[8], q[16];
rz(-0.7197777032852173) q[16];
cx q[8], q[16];
cx q[25], q[4];
rz(-0.7197777032852173) q[4];
cx q[25], q[4];
cx q[25], q[15];
rz(-0.7197777032852173) q[15];
cx q[25], q[15];
cx q[30], q[3];
rz(-0.7197777032852173) q[3];
cx q[30], q[3];
cx q[9], q[17];
rz(-0.7197777032852173) q[17];
cx q[9], q[17];
cx q[26], q[23];
rz(-0.7197777032852173) q[23];
cx q[26], q[23];
cx q[16], q[19];
rz(-0.7197777032852173) q[19];
cx q[16], q[19];
cx q[25], q[23];
rz(-0.7197777032852173) q[23];
cx q[25], q[23];
cx q[21], q[6];
rz(-0.7197777032852173) q[6];
cx q[21], q[6];
cx q[28], q[22];
rz(-0.7197777032852173) q[22];
cx q[28], q[22];
cx q[29], q[16];
rz(-0.7197777032852173) q[16];
cx q[29], q[16];
cx q[3], q[15];
rz(-0.7197777032852173) q[15];
cx q[3], q[15];
cx q[17], q[24];
rz(-0.7197777032852173) q[24];
cx q[17], q[24];
cx q[26], q[15];
rz(-0.7197777032852173) q[15];
cx q[26], q[15];
cx q[5], q[2];
rz(-0.7197777032852173) q[2];
cx q[5], q[2];
cx q[16], q[22];
rz(-0.7197777032852173) q[22];
cx q[16], q[22];
cx q[6], q[2];
rz(-0.7197777032852173) q[2];
cx q[6], q[2];
cx q[9], q[15];
rz(-0.7197777032852173) q[15];
cx q[9], q[15];
cx q[20], q[17];
rz(-0.7197777032852173) q[17];
cx q[20], q[17];
cx q[4], q[14];
rz(-0.7197777032852173) q[14];
cx q[4], q[14];
cx q[9], q[30];
rz(-0.7197777032852173) q[30];
cx q[9], q[30];
cx q[2], q[9];
rz(-0.7197777032852173) q[9];
cx q[2], q[9];
cx q[29], q[14];
rz(-0.7197777032852173) q[14];
cx q[29], q[14];
cx q[16], q[28];
rz(-0.7197777032852173) q[28];
cx q[16], q[28];
cx q[22], q[10];
rz(-0.7197777032852173) q[10];
cx q[22], q[10];
cx q[3], q[4];
rz(-0.7197777032852173) q[4];
cx q[3], q[4];
cx q[30], q[11];
rz(-0.7197777032852173) q[11];
cx q[30], q[11];
cx q[19], q[2];
rz(-0.7197777032852173) q[2];
cx q[19], q[2];
cx q[25], q[19];
rz(-0.7197777032852173) q[19];
cx q[25], q[19];
cx q[13], q[14];
rz(-0.7197777032852173) q[14];
cx q[13], q[14];
cx q[16], q[21];
rz(-0.7197777032852173) q[21];
cx q[16], q[21];
cx q[2], q[12];
rz(-0.7197777032852173) q[12];
cx q[2], q[12];
cx q[5], q[25];
rz(-0.7197777032852173) q[25];
cx q[5], q[25];
cx q[2], q[24];
rz(-0.7197777032852173) q[24];
cx q[2], q[24];
cx q[27], q[11];
rz(-0.7197777032852173) q[11];
cx q[27], q[11];
cx q[7], q[16];
rz(-0.7197777032852173) q[16];
cx q[7], q[16];
cx q[16], q[5];
rz(-0.7197777032852173) q[5];
cx q[16], q[5];
cx q[19], q[12];
rz(-0.7197777032852173) q[12];
cx q[19], q[12];
cx q[10], q[3];
rz(-0.7197777032852173) q[3];
cx q[10], q[3];
cx q[0], q[29];
rz(-0.7197777032852173) q[29];
cx q[0], q[29];
cx q[12], q[0];
rz(-0.7197777032852173) q[0];
cx q[12], q[0];
cx q[10], q[25];
rz(-0.7197777032852173) q[25];
cx q[10], q[25];
cx q[30], q[10];
rz(-0.7197777032852173) q[10];
cx q[30], q[10];
cx q[10], q[9];
rz(-0.7197777032852173) q[9];
cx q[10], q[9];
cx q[2], q[25];
rz(-0.7197777032852173) q[25];
cx q[2], q[25];
cx q[28], q[27];
rz(-0.7197777032852173) q[27];
cx q[28], q[27];
cx q[19], q[13];
rz(-0.7197777032852173) q[13];
cx q[19], q[13];
cx q[3], q[17];
rz(-0.7197777032852173) q[17];
cx q[3], q[17];
cx q[5], q[0];
rz(-0.7197777032852173) q[0];
cx q[5], q[0];
cx q[21], q[2];
rz(-0.7197777032852173) q[2];
cx q[21], q[2];
cx q[10], q[16];
rz(-0.7197777032852173) q[16];
cx q[10], q[16];
cx q[19], q[5];
rz(-0.7197777032852173) q[5];
cx q[19], q[5];
cx q[3], q[11];
rz(-0.7197777032852173) q[11];
cx q[3], q[11];
cx q[20], q[28];
rz(-0.7197777032852173) q[28];
cx q[20], q[28];
cx q[20], q[10];
rz(-0.7197777032852173) q[10];
cx q[20], q[10];
cx q[19], q[29];
rz(-0.7197777032852173) q[29];
cx q[19], q[29];
cx q[30], q[16];
rz(-0.7197777032852173) q[16];
cx q[30], q[16];
cx q[5], q[29];
rz(-0.7197777032852173) q[29];
cx q[5], q[29];
cx q[9], q[31];
rz(-0.7197777032852173) q[31];
cx q[9], q[31];
cx q[17], q[25];
rz(-0.7197777032852173) q[25];
cx q[17], q[25];
cx q[0], q[20];
rz(-0.7197777032852173) q[20];
cx q[0], q[20];
cx q[31], q[13];
rz(-0.7197777032852173) q[13];
cx q[31], q[13];
cx q[14], q[6];
rz(-0.7197777032852173) q[6];
cx q[14], q[6];
cx q[7], q[1];
rz(-0.7197777032852173) q[1];
cx q[7], q[1];
cx q[24], q[5];
rz(-0.7197777032852173) q[5];
cx q[24], q[5];
cx q[5], q[20];
rz(-0.7197777032852173) q[20];
cx q[5], q[20];
cx q[6], q[31];
rz(-0.7197777032852173) q[31];
cx q[6], q[31];
cx q[14], q[3];
rz(-0.7197777032852173) q[3];
cx q[14], q[3];
cx q[15], q[20];
rz(-0.7197777032852173) q[20];
cx q[15], q[20];
cx q[29], q[3];
rz(-0.7197777032852173) q[3];
cx q[29], q[3];
cx q[31], q[8];
rz(-0.7197777032852173) q[8];
cx q[31], q[8];
cx q[14], q[18];
rz(-0.7197777032852173) q[18];
cx q[14], q[18];
cx q[29], q[4];
rz(-0.7197777032852173) q[4];
cx q[29], q[4];
cx q[31], q[18];
rz(-0.7197777032852173) q[18];
cx q[31], q[18];
cx q[30], q[23];
rz(-0.7197777032852173) q[23];
cx q[30], q[23];
cx q[9], q[24];
rz(-0.7197777032852173) q[24];
cx q[9], q[24];
cx q[1], q[2];
rz(-0.7197777032852173) q[2];
cx q[1], q[2];
cx q[10], q[24];
rz(-0.7197777032852173) q[24];
cx q[10], q[24];
cx q[5], q[14];
rz(-0.7197777032852173) q[14];
cx q[5], q[14];
cx q[27], q[22];
rz(-0.7197777032852173) q[22];
cx q[27], q[22];
cx q[0], q[1];
rz(-0.7197777032852173) q[1];
cx q[0], q[1];
cx q[20], q[22];
rz(-0.7197777032852173) q[22];
cx q[20], q[22];
cx q[11], q[1];
rz(-0.7197777032852173) q[1];
cx q[11], q[1];
cx q[1], q[16];
rz(-0.7197777032852173) q[16];
cx q[1], q[16];
cx q[10], q[5];
rz(-0.7197777032852173) q[5];
cx q[10], q[5];
cx q[29], q[27];
rz(-0.7197777032852173) q[27];
cx q[29], q[27];
cx q[30], q[12];
rz(-0.7197777032852173) q[12];
cx q[30], q[12];
cx q[23], q[7];
rz(-0.7197777032852173) q[7];
cx q[23], q[7];
cx q[30], q[18];
rz(-0.7197777032852173) q[18];
cx q[30], q[18];
cx q[17], q[10];
rz(-0.7197777032852173) q[10];
cx q[17], q[10];
cx q[31], q[17];
rz(-0.7197777032852173) q[17];
cx q[31], q[17];
cx q[2], q[15];
rz(-0.7197777032852173) q[15];
cx q[2], q[15];
cx q[3], q[8];
rz(-0.7197777032852173) q[8];
cx q[3], q[8];
cx q[24], q[13];
rz(-0.7197777032852173) q[13];
cx q[24], q[13];
cx q[22], q[6];
rz(-0.7197777032852173) q[6];
cx q[22], q[6];
cx q[6], q[1];
rz(-0.7197777032852173) q[1];
cx q[6], q[1];
cx q[14], q[24];
rz(-0.7197777032852173) q[24];
cx q[14], q[24];
cx q[22], q[5];
rz(-0.7197777032852173) q[5];
cx q[22], q[5];
cx q[24], q[23];
rz(-0.7197777032852173) q[23];
cx q[24], q[23];
cx q[28], q[29];
rz(-0.7197777032852173) q[29];
cx q[28], q[29];
cx q[11], q[26];
rz(-0.7197777032852173) q[26];
cx q[11], q[26];
cx q[12], q[14];
rz(-0.7197777032852173) q[14];
cx q[12], q[14];
cx q[11], q[31];
rz(-0.7197777032852173) q[31];
cx q[11], q[31];
cx q[16], q[25];
rz(-0.7197777032852173) q[25];
cx q[16], q[25];
cx q[10], q[27];
rz(-0.7197777032852173) q[27];
cx q[10], q[27];
cx q[20], q[25];
rz(-0.7197777032852173) q[25];
cx q[20], q[25];
cx q[8], q[11];
rz(-0.7197777032852173) q[11];
cx q[8], q[11];
cx q[11], q[20];
rz(-0.7197777032852173) q[20];
cx q[11], q[20];
cx q[22], q[12];
rz(-0.7197777032852173) q[12];
cx q[22], q[12];
cx q[22], q[21];
rz(-0.7197777032852173) q[21];
cx q[22], q[21];
cx q[17], q[29];
rz(-0.7197777032852173) q[29];
cx q[17], q[29];
cx q[5], q[13];
rz(-0.7197777032852173) q[13];
cx q[5], q[13];
cx q[24], q[21];
rz(-0.7197777032852173) q[21];
cx q[24], q[21];
cx q[26], q[31];
rz(-0.7197777032852173) q[31];
cx q[26], q[31];
cx q[23], q[6];
rz(-0.7197777032852173) q[6];
cx q[23], q[6];
cx q[13], q[25];
rz(-0.7197777032852173) q[25];
cx q[13], q[25];
cx q[22], q[13];
rz(-0.7197777032852173) q[13];
cx q[22], q[13];
cx q[8], q[2];
rz(-0.7197777032852173) q[2];
cx q[8], q[2];
cx q[6], q[20];
rz(-0.7197777032852173) q[20];
cx q[6], q[20];
cx q[4], q[23];
rz(-0.7197777032852173) q[23];
cx q[4], q[23];
cx q[29], q[8];
rz(-0.7197777032852173) q[8];
cx q[29], q[8];
cx q[10], q[13];
rz(-0.7197777032852173) q[13];
cx q[10], q[13];
cx q[29], q[24];
rz(-0.7197777032852173) q[24];
cx q[29], q[24];
cx q[21], q[28];
rz(-0.7197777032852173) q[28];
cx q[21], q[28];
cx q[7], q[9];
rz(-0.7197777032852173) q[9];
cx q[7], q[9];
cx q[13], q[1];
rz(-0.7197777032852173) q[1];
cx q[13], q[1];
cx q[4], q[28];
rz(-0.7197777032852173) q[28];
cx q[4], q[28];
cx q[3], q[22];
rz(-0.7197777032852173) q[22];
cx q[3], q[22];
cx q[6], q[19];
rz(-0.7197777032852173) q[19];
cx q[6], q[19];
cx q[10], q[7];
rz(-0.7197777032852173) q[7];
cx q[10], q[7];
cx q[0], q[8];
rz(-0.7197777032852173) q[8];
cx q[0], q[8];
cx q[26], q[22];
rz(-0.7197777032852173) q[22];
cx q[26], q[22];
cx q[31], q[27];
rz(-0.7197777032852173) q[27];
cx q[31], q[27];
cx q[9], q[12];
rz(-0.7197777032852173) q[12];
cx q[9], q[12];
cx q[29], q[21];
rz(-0.7197777032852173) q[21];
cx q[29], q[21];
cx q[4], q[21];
rz(-0.7197777032852173) q[21];
cx q[4], q[21];
cx q[13], q[8];
rz(-0.7197777032852173) q[8];
cx q[13], q[8];
cx q[1], q[9];
rz(-0.7197777032852173) q[9];
cx q[1], q[9];
cx q[15], q[4];
rz(-0.7197777032852173) q[4];
cx q[15], q[4];
cx q[17], q[8];
rz(-0.7197777032852173) q[8];
cx q[17], q[8];
cx q[24], q[7];
rz(-0.7197777032852173) q[7];
cx q[24], q[7];
cx q[17], q[15];
rz(-0.7197777032852173) q[15];
cx q[17], q[15];
cx q[28], q[15];
rz(-0.7197777032852173) q[15];
cx q[28], q[15];
cx q[30], q[20];
rz(-0.7197777032852173) q[20];
cx q[30], q[20];
cx q[26], q[2];
rz(-0.7197777032852173) q[2];
cx q[26], q[2];
cx q[14], q[25];
rz(-0.7197777032852173) q[25];
cx q[14], q[25];
cx q[17], q[11];
rz(-0.7197777032852173) q[11];
cx q[17], q[11];
cx q[0], q[19];
rz(-0.7197777032852173) q[19];
cx q[0], q[19];
cx q[3], q[31];
rz(-0.7197777032852173) q[31];
cx q[3], q[31];
cx q[24], q[6];
rz(-0.7197777032852173) q[6];
cx q[24], q[6];
cx q[26], q[0];
rz(-0.7197777032852173) q[0];
cx q[26], q[0];
cx q[31], q[22];
rz(-0.7197777032852173) q[22];
cx q[31], q[22];
cx q[23], q[2];
rz(-0.7197777032852173) q[2];
cx q[23], q[2];
cx q[16], q[15];
rz(-0.7197777032852173) q[15];
cx q[16], q[15];
cx q[5], q[28];
rz(-0.7197777032852173) q[28];
cx q[5], q[28];
cx q[8], q[10];
rz(-0.7197777032852173) q[10];
cx q[8], q[10];
cx q[27], q[0];
rz(-0.7197777032852173) q[0];
cx q[27], q[0];
cx q[30], q[27];
rz(-0.7197777032852173) q[27];
cx q[30], q[27];
cx q[18], q[10];
rz(-0.7197777032852173) q[10];
cx q[18], q[10];
cx q[13], q[17];
rz(-0.7197777032852173) q[17];
cx q[13], q[17];
cx q[7], q[4];
rz(-0.7197777032852173) q[4];
cx q[7], q[4];
cx q[4], q[27];
rz(-0.7197777032852173) q[27];
cx q[4], q[27];
cx q[27], q[1];
rz(-0.7197777032852173) q[1];
cx q[27], q[1];
cx q[14], q[23];
rz(-0.7197777032852173) q[23];
cx q[14], q[23];
cx q[20], q[4];
rz(-0.7197777032852173) q[4];
cx q[20], q[4];
cx q[9], q[3];
rz(-0.7197777032852173) q[3];
cx q[9], q[3];
cx q[8], q[19];
rz(-0.7197777032852173) q[19];
cx q[8], q[19];
cx q[14], q[22];
rz(-0.7197777032852173) q[22];
cx q[14], q[22];
cx q[8], q[27];
rz(-0.7197777032852173) q[27];
cx q[8], q[27];
cx q[11], q[24];
rz(-0.7197777032852173) q[24];
cx q[11], q[24];
cx q[20], q[26];
rz(-0.7197777032852173) q[26];
cx q[20], q[26];
cx q[4], q[30];
rz(-0.7197777032852173) q[30];
cx q[4], q[30];
cx q[26], q[17];
rz(-0.7197777032852173) q[17];
cx q[26], q[17];
cx q[15], q[13];
rz(-0.7197777032852173) q[13];
cx q[15], q[13];
cx q[24], q[25];
rz(-0.7197777032852173) q[25];
cx q[24], q[25];
cx q[25], q[1];
rz(-0.7197777032852173) q[1];
cx q[25], q[1];
cx q[31], q[2];
rz(-0.7197777032852173) q[2];
cx q[31], q[2];
cx q[26], q[5];
rz(-0.7197777032852173) q[5];
cx q[26], q[5];
cx q[28], q[2];
rz(-0.7197777032852173) q[2];
cx q[28], q[2];
cx q[16], q[24];
rz(-0.7197777032852173) q[24];
cx q[16], q[24];
cx q[18], q[11];
rz(-0.7197777032852173) q[11];
cx q[18], q[11];
cx q[9], q[16];
rz(-0.7197777032852173) q[16];
cx q[9], q[16];
cx q[21], q[30];
rz(-0.7197777032852173) q[30];
cx q[21], q[30];
cx q[21], q[0];
rz(-0.7197777032852173) q[0];
cx q[21], q[0];
cx q[26], q[1];
rz(-0.7197777032852173) q[1];
cx q[26], q[1];
cx q[15], q[0];
rz(-0.7197777032852173) q[0];
cx q[15], q[0];
cx q[23], q[18];
rz(-0.7197777032852173) q[18];
cx q[23], q[18];
cx q[13], q[26];
rz(-0.7197777032852173) q[26];
cx q[13], q[26];
cx q[5], q[27];
rz(-0.7197777032852173) q[27];
cx q[5], q[27];
cx q[6], q[13];
rz(-0.7197777032852173) q[13];
cx q[6], q[13];
cx q[14], q[0];
rz(-0.7197777032852173) q[0];
cx q[14], q[0];
cx q[18], q[8];
rz(-0.7197777032852173) q[8];
cx q[18], q[8];
cx q[5], q[23];
rz(-0.7197777032852173) q[23];
cx q[5], q[23];
cx q[11], q[23];
rz(-0.7197777032852173) q[23];
cx q[11], q[23];
cx q[11], q[2];
rz(-0.7197777032852173) q[2];
cx q[11], q[2];
cx q[21], q[15];
rz(-0.7197777032852173) q[15];
cx q[21], q[15];
cx q[13], q[23];
rz(-0.7197777032852173) q[23];
cx q[13], q[23];
cx q[26], q[12];
rz(-0.7197777032852173) q[12];
cx q[26], q[12];
cx q[3], q[16];
rz(-0.7197777032852173) q[16];
cx q[3], q[16];
cx q[25], q[6];
rz(-0.7197777032852173) q[6];
cx q[25], q[6];
cx q[6], q[8];
rz(-0.7197777032852173) q[8];
cx q[6], q[8];
cx q[1], q[12];
rz(-0.7197777032852173) q[12];
cx q[1], q[12];
cx q[15], q[5];
rz(-0.7197777032852173) q[5];
cx q[15], q[5];
cx q[12], q[17];
rz(-0.7197777032852173) q[17];
cx q[12], q[17];
cx q[18], q[29];
rz(-0.7197777032852173) q[29];
cx q[18], q[29];
cx q[9], q[22];
rz(-0.7197777032852173) q[22];
cx q[9], q[22];
cx q[30], q[5];
rz(-0.7197777032852173) q[5];
cx q[30], q[5];
cx q[16], q[31];
rz(-0.7197777032852173) q[31];
cx q[16], q[31];
cx q[22], q[4];
rz(-0.7197777032852173) q[4];
cx q[22], q[4];
cx q[7], q[3];
rz(-0.7197777032852173) q[3];
cx q[7], q[3];
cx q[23], q[28];
rz(-0.7197777032852173) q[28];
cx q[23], q[28];
cx q[6], q[12];
rz(-0.7197777032852173) q[12];
cx q[6], q[12];
cx q[5], q[3];
rz(-0.7197777032852173) q[3];
cx q[5], q[3];
cx q[9], q[13];
rz(-0.7197777032852173) q[13];
cx q[9], q[13];
cx q[26], q[29];
rz(-0.7197777032852173) q[29];
cx q[26], q[29];
cx q[17], q[18];
rz(-0.7197777032852173) q[18];
cx q[17], q[18];
cx q[0], q[7];
rz(-0.7197777032852173) q[7];
cx q[0], q[7];
cx q[27], q[14];
rz(-0.7197777032852173) q[14];
cx q[27], q[14];
cx q[11], q[25];
rz(-0.7197777032852173) q[25];
cx q[11], q[25];
cx q[4], q[5];
rz(-0.7197777032852173) q[5];
cx q[4], q[5];
cx q[20], q[21];
rz(-0.7197777032852173) q[21];
cx q[20], q[21];
cx q[7], q[30];
rz(-0.7197777032852173) q[30];
cx q[7], q[30];
cx q[17], q[28];
rz(-0.7197777032852173) q[28];
cx q[17], q[28];
cx q[29], q[1];
rz(-0.7197777032852173) q[1];
cx q[29], q[1];
cx q[7], q[22];
rz(-0.7197777032852173) q[22];
cx q[7], q[22];
cx q[30], q[22];
rz(-0.7197777032852173) q[22];
cx q[30], q[22];
cx q[26], q[24];
rz(-0.7197777032852173) q[24];
cx q[26], q[24];
cx q[11], q[29];
rz(-0.7197777032852173) q[29];
cx q[11], q[29];
cx q[19], q[15];
rz(-0.7197777032852173) q[15];
cx q[19], q[15];
cx q[23], q[10];
rz(-0.7197777032852173) q[10];
cx q[23], q[10];
cx q[13], q[0];
rz(-0.7197777032852173) q[0];
cx q[13], q[0];
cx q[16], q[23];
rz(-0.7197777032852173) q[23];
cx q[16], q[23];
cx q[28], q[3];
rz(-0.7197777032852173) q[3];
cx q[28], q[3];
cx q[1], q[28];
rz(-0.7197777032852173) q[28];
cx q[1], q[28];
cx q[30], q[25];
rz(-0.7197777032852173) q[25];
cx q[30], q[25];
cx q[16], q[14];
rz(-0.7197777032852173) q[14];
cx q[16], q[14];
cx q[18], q[4];
rz(-0.7197777032852173) q[4];
cx q[18], q[4];
cx q[1], q[19];
rz(-0.7197777032852173) q[19];
cx q[1], q[19];
cx q[3], q[23];
rz(-0.7197777032852173) q[23];
cx q[3], q[23];
cx q[23], q[31];
rz(-0.7197777032852173) q[31];
cx q[23], q[31];
cx q[1], q[14];
rz(-0.7197777032852173) q[14];
cx q[1], q[14];
cx q[21], q[3];
rz(-0.7197777032852173) q[3];
cx q[21], q[3];
cx q[0], q[30];
rz(-0.7197777032852173) q[30];
cx q[0], q[30];
cx q[10], q[2];
rz(-0.7197777032852173) q[2];
cx q[10], q[2];
cx q[0], q[10];
rz(-0.7197777032852173) q[10];
cx q[0], q[10];
rx(0.6278608441352844) q[0];
rx(0.6278608441352844) q[1];
rx(0.6278608441352844) q[2];
rx(0.6278608441352844) q[3];
rx(0.6278608441352844) q[4];
rx(0.6278608441352844) q[5];
rx(0.6278608441352844) q[6];
rx(0.6278608441352844) q[7];
rx(0.6278608441352844) q[8];
rx(0.6278608441352844) q[9];
rx(0.6278608441352844) q[10];
rx(0.6278608441352844) q[11];
rx(0.6278608441352844) q[12];
rx(0.6278608441352844) q[13];
rx(0.6278608441352844) q[14];
rx(0.6278608441352844) q[15];
rx(0.6278608441352844) q[16];
rx(0.6278608441352844) q[17];
rx(0.6278608441352844) q[18];
rx(0.6278608441352844) q[19];
rx(0.6278608441352844) q[20];
rx(0.6278608441352844) q[21];
rx(0.6278608441352844) q[22];
rx(0.6278608441352844) q[23];
rx(0.6278608441352844) q[24];
rx(0.6278608441352844) q[25];
rx(0.6278608441352844) q[26];
rx(0.6278608441352844) q[27];
rx(0.6278608441352844) q[28];
rx(0.6278608441352844) q[29];
rx(0.6278608441352844) q[30];
rx(0.6278608441352844) q[31];