OPENQASM 2.0;
include "qelib1.inc";
qreg q[47];
creg c[47];
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
h q[32];
h q[33];
h q[34];
h q[35];
h q[36];
h q[37];
h q[38];
h q[39];
h q[40];
h q[41];
h q[42];
h q[43];
h q[44];
h q[45];
h q[46];
cx q[8], q[4];
rz(-0.2542710304260254) q[4];
cx q[8], q[4];
cx q[43], q[4];
rz(-0.2542710304260254) q[4];
cx q[43], q[4];
cx q[7], q[5];
rz(-0.2542710304260254) q[5];
cx q[7], q[5];
cx q[34], q[45];
rz(-0.2542710304260254) q[45];
cx q[34], q[45];
cx q[22], q[41];
rz(-0.2542710304260254) q[41];
cx q[22], q[41];
cx q[34], q[37];
rz(-0.2542710304260254) q[37];
cx q[34], q[37];
cx q[8], q[9];
rz(-0.2542710304260254) q[9];
cx q[8], q[9];
cx q[14], q[26];
rz(-0.2542710304260254) q[26];
cx q[14], q[26];
cx q[41], q[45];
rz(-0.2542710304260254) q[45];
cx q[41], q[45];
cx q[10], q[39];
rz(-0.2542710304260254) q[39];
cx q[10], q[39];
cx q[5], q[19];
rz(-0.2542710304260254) q[19];
cx q[5], q[19];
cx q[41], q[0];
rz(-0.2542710304260254) q[0];
cx q[41], q[0];
cx q[5], q[21];
rz(-0.2542710304260254) q[21];
cx q[5], q[21];
cx q[6], q[5];
rz(-0.2542710304260254) q[5];
cx q[6], q[5];
cx q[36], q[10];
rz(-0.2542710304260254) q[10];
cx q[36], q[10];
cx q[30], q[32];
rz(-0.2542710304260254) q[32];
cx q[30], q[32];
cx q[37], q[9];
rz(-0.2542710304260254) q[9];
cx q[37], q[9];
cx q[35], q[25];
rz(-0.2542710304260254) q[25];
cx q[35], q[25];
cx q[21], q[45];
rz(-0.2542710304260254) q[45];
cx q[21], q[45];
cx q[46], q[24];
rz(-0.2542710304260254) q[24];
cx q[46], q[24];
cx q[26], q[1];
rz(-0.2542710304260254) q[1];
cx q[26], q[1];
cx q[33], q[2];
rz(-0.2542710304260254) q[2];
cx q[33], q[2];
cx q[26], q[37];
rz(-0.2542710304260254) q[37];
cx q[26], q[37];
cx q[2], q[19];
rz(-0.2542710304260254) q[19];
cx q[2], q[19];
cx q[40], q[26];
rz(-0.2542710304260254) q[26];
cx q[40], q[26];
cx q[32], q[23];
rz(-0.2542710304260254) q[23];
cx q[32], q[23];
cx q[31], q[13];
rz(-0.2542710304260254) q[13];
cx q[31], q[13];
cx q[36], q[42];
rz(-0.2542710304260254) q[42];
cx q[36], q[42];
cx q[0], q[11];
rz(-0.2542710304260254) q[11];
cx q[0], q[11];
cx q[17], q[12];
rz(-0.2542710304260254) q[12];
cx q[17], q[12];
cx q[42], q[30];
rz(-0.2542710304260254) q[30];
cx q[42], q[30];
cx q[40], q[10];
rz(-0.2542710304260254) q[10];
cx q[40], q[10];
cx q[43], q[3];
rz(-0.2542710304260254) q[3];
cx q[43], q[3];
cx q[45], q[15];
rz(-0.2542710304260254) q[15];
cx q[45], q[15];
cx q[33], q[39];
rz(-0.2542710304260254) q[39];
cx q[33], q[39];
cx q[29], q[45];
rz(-0.2542710304260254) q[45];
cx q[29], q[45];
cx q[6], q[4];
rz(-0.2542710304260254) q[4];
cx q[6], q[4];
cx q[41], q[20];
rz(-0.2542710304260254) q[20];
cx q[41], q[20];
cx q[5], q[12];
rz(-0.2542710304260254) q[12];
cx q[5], q[12];
cx q[37], q[36];
rz(-0.2542710304260254) q[36];
cx q[37], q[36];
cx q[24], q[5];
rz(-0.2542710304260254) q[5];
cx q[24], q[5];
cx q[27], q[35];
rz(-0.2542710304260254) q[35];
cx q[27], q[35];
cx q[27], q[26];
rz(-0.2542710304260254) q[26];
cx q[27], q[26];
cx q[12], q[29];
rz(-0.2542710304260254) q[29];
cx q[12], q[29];
cx q[12], q[24];
rz(-0.2542710304260254) q[24];
cx q[12], q[24];
cx q[46], q[34];
rz(-0.2542710304260254) q[34];
cx q[46], q[34];
cx q[13], q[14];
rz(-0.2542710304260254) q[14];
cx q[13], q[14];
cx q[8], q[43];
rz(-0.2542710304260254) q[43];
cx q[8], q[43];
cx q[5], q[38];
rz(-0.2542710304260254) q[38];
cx q[5], q[38];
cx q[7], q[22];
rz(-0.2542710304260254) q[22];
cx q[7], q[22];
cx q[36], q[27];
rz(-0.2542710304260254) q[27];
cx q[36], q[27];
cx q[36], q[31];
rz(-0.2542710304260254) q[31];
cx q[36], q[31];
cx q[17], q[42];
rz(-0.2542710304260254) q[42];
cx q[17], q[42];
cx q[0], q[10];
rz(-0.2542710304260254) q[10];
cx q[0], q[10];
cx q[33], q[15];
rz(-0.2542710304260254) q[15];
cx q[33], q[15];
cx q[3], q[45];
rz(-0.2542710304260254) q[45];
cx q[3], q[45];
cx q[4], q[37];
rz(-0.2542710304260254) q[37];
cx q[4], q[37];
cx q[33], q[21];
rz(-0.2542710304260254) q[21];
cx q[33], q[21];
cx q[18], q[2];
rz(-0.2542710304260254) q[2];
cx q[18], q[2];
cx q[1], q[13];
rz(-0.2542710304260254) q[13];
cx q[1], q[13];
cx q[35], q[43];
rz(-0.2542710304260254) q[43];
cx q[35], q[43];
cx q[21], q[15];
rz(-0.2542710304260254) q[15];
cx q[21], q[15];
cx q[24], q[14];
rz(-0.2542710304260254) q[14];
cx q[24], q[14];
cx q[17], q[22];
rz(-0.2542710304260254) q[22];
cx q[17], q[22];
cx q[20], q[14];
rz(-0.2542710304260254) q[14];
cx q[20], q[14];
cx q[42], q[13];
rz(-0.2542710304260254) q[13];
cx q[42], q[13];
cx q[22], q[16];
rz(-0.2542710304260254) q[16];
cx q[22], q[16];
cx q[9], q[7];
rz(-0.2542710304260254) q[7];
cx q[9], q[7];
cx q[41], q[46];
rz(-0.2542710304260254) q[46];
cx q[41], q[46];
cx q[39], q[38];
rz(-0.2542710304260254) q[38];
cx q[39], q[38];
cx q[4], q[1];
rz(-0.2542710304260254) q[1];
cx q[4], q[1];
cx q[26], q[8];
rz(-0.2542710304260254) q[8];
cx q[26], q[8];
cx q[11], q[40];
rz(-0.2542710304260254) q[40];
cx q[11], q[40];
cx q[7], q[40];
rz(-0.2542710304260254) q[40];
cx q[7], q[40];
cx q[22], q[2];
rz(-0.2542710304260254) q[2];
cx q[22], q[2];
cx q[8], q[42];
rz(-0.2542710304260254) q[42];
cx q[8], q[42];
cx q[28], q[13];
rz(-0.2542710304260254) q[13];
cx q[28], q[13];
cx q[32], q[31];
rz(-0.2542710304260254) q[31];
cx q[32], q[31];
cx q[14], q[42];
rz(-0.2542710304260254) q[42];
cx q[14], q[42];
cx q[14], q[1];
rz(-0.2542710304260254) q[1];
cx q[14], q[1];
cx q[23], q[1];
rz(-0.2542710304260254) q[1];
cx q[23], q[1];
cx q[26], q[39];
rz(-0.2542710304260254) q[39];
cx q[26], q[39];
cx q[2], q[39];
rz(-0.2542710304260254) q[39];
cx q[2], q[39];
cx q[4], q[16];
rz(-0.2542710304260254) q[16];
cx q[4], q[16];
cx q[19], q[14];
rz(-0.2542710304260254) q[14];
cx q[19], q[14];
cx q[40], q[1];
rz(-0.2542710304260254) q[1];
cx q[40], q[1];
cx q[16], q[8];
rz(-0.2542710304260254) q[8];
cx q[16], q[8];
cx q[4], q[10];
rz(-0.2542710304260254) q[10];
cx q[4], q[10];
cx q[29], q[26];
rz(-0.2542710304260254) q[26];
cx q[29], q[26];
cx q[35], q[9];
rz(-0.2542710304260254) q[9];
cx q[35], q[9];
cx q[34], q[7];
rz(-0.2542710304260254) q[7];
cx q[34], q[7];
cx q[40], q[6];
rz(-0.2542710304260254) q[6];
cx q[40], q[6];
cx q[28], q[44];
rz(-0.2542710304260254) q[44];
cx q[28], q[44];
cx q[38], q[36];
rz(-0.2542710304260254) q[36];
cx q[38], q[36];
cx q[46], q[37];
rz(-0.2542710304260254) q[37];
cx q[46], q[37];
cx q[15], q[13];
rz(-0.2542710304260254) q[13];
cx q[15], q[13];
cx q[12], q[26];
rz(-0.2542710304260254) q[26];
cx q[12], q[26];
cx q[28], q[30];
rz(-0.2542710304260254) q[30];
cx q[28], q[30];
cx q[15], q[41];
rz(-0.2542710304260254) q[41];
cx q[15], q[41];
cx q[17], q[21];
rz(-0.2542710304260254) q[21];
cx q[17], q[21];
cx q[32], q[38];
rz(-0.2542710304260254) q[38];
cx q[32], q[38];
cx q[6], q[9];
rz(-0.2542710304260254) q[9];
cx q[6], q[9];
cx q[31], q[9];
rz(-0.2542710304260254) q[9];
cx q[31], q[9];
cx q[14], q[7];
rz(-0.2542710304260254) q[7];
cx q[14], q[7];
cx q[22], q[37];
rz(-0.2542710304260254) q[37];
cx q[22], q[37];
cx q[33], q[34];
rz(-0.2542710304260254) q[34];
cx q[33], q[34];
cx q[13], q[23];
rz(-0.2542710304260254) q[23];
cx q[13], q[23];
cx q[34], q[22];
rz(-0.2542710304260254) q[22];
cx q[34], q[22];
cx q[40], q[29];
rz(-0.2542710304260254) q[29];
cx q[40], q[29];
cx q[32], q[15];
rz(-0.2542710304260254) q[15];
cx q[32], q[15];
cx q[34], q[21];
rz(-0.2542710304260254) q[21];
cx q[34], q[21];
cx q[22], q[23];
rz(-0.2542710304260254) q[23];
cx q[22], q[23];
cx q[20], q[21];
rz(-0.2542710304260254) q[21];
cx q[20], q[21];
cx q[3], q[42];
rz(-0.2542710304260254) q[42];
cx q[3], q[42];
cx q[39], q[21];
rz(-0.2542710304260254) q[21];
cx q[39], q[21];
cx q[33], q[17];
rz(-0.2542710304260254) q[17];
cx q[33], q[17];
cx q[36], q[40];
rz(-0.2542710304260254) q[40];
cx q[36], q[40];
cx q[42], q[1];
rz(-0.2542710304260254) q[1];
cx q[42], q[1];
cx q[15], q[28];
rz(-0.2542710304260254) q[28];
cx q[15], q[28];
cx q[45], q[2];
rz(-0.2542710304260254) q[2];
cx q[45], q[2];
cx q[44], q[31];
rz(-0.2542710304260254) q[31];
cx q[44], q[31];
cx q[46], q[30];
rz(-0.2542710304260254) q[30];
cx q[46], q[30];
cx q[39], q[43];
rz(-0.2542710304260254) q[43];
cx q[39], q[43];
cx q[2], q[38];
rz(-0.2542710304260254) q[38];
cx q[2], q[38];
cx q[36], q[43];
rz(-0.2542710304260254) q[43];
cx q[36], q[43];
cx q[0], q[25];
rz(-0.2542710304260254) q[25];
cx q[0], q[25];
cx q[28], q[36];
rz(-0.2542710304260254) q[36];
cx q[28], q[36];
cx q[33], q[32];
rz(-0.2542710304260254) q[32];
cx q[33], q[32];
cx q[16], q[23];
rz(-0.2542710304260254) q[23];
cx q[16], q[23];
cx q[16], q[3];
rz(-0.2542710304260254) q[3];
cx q[16], q[3];
cx q[21], q[38];
rz(-0.2542710304260254) q[38];
cx q[21], q[38];
cx q[42], q[15];
rz(-0.2542710304260254) q[15];
cx q[42], q[15];
cx q[11], q[44];
rz(-0.2542710304260254) q[44];
cx q[11], q[44];
cx q[27], q[17];
rz(-0.2542710304260254) q[17];
cx q[27], q[17];
cx q[22], q[9];
rz(-0.2542710304260254) q[9];
cx q[22], q[9];
cx q[5], q[2];
rz(-0.2542710304260254) q[2];
cx q[5], q[2];
cx q[3], q[18];
rz(-0.2542710304260254) q[18];
cx q[3], q[18];
cx q[15], q[3];
rz(-0.2542710304260254) q[3];
cx q[15], q[3];
cx q[1], q[32];
rz(-0.2542710304260254) q[32];
cx q[1], q[32];
cx q[8], q[44];
rz(-0.2542710304260254) q[44];
cx q[8], q[44];
cx q[1], q[31];
rz(-0.2542710304260254) q[31];
cx q[1], q[31];
cx q[41], q[26];
rz(-0.2542710304260254) q[26];
cx q[41], q[26];
cx q[20], q[10];
rz(-0.2542710304260254) q[10];
cx q[20], q[10];
cx q[20], q[3];
rz(-0.2542710304260254) q[3];
cx q[20], q[3];
cx q[14], q[33];
rz(-0.2542710304260254) q[33];
cx q[14], q[33];
cx q[26], q[9];
rz(-0.2542710304260254) q[9];
cx q[26], q[9];
cx q[6], q[23];
rz(-0.2542710304260254) q[23];
cx q[6], q[23];
cx q[27], q[38];
rz(-0.2542710304260254) q[38];
cx q[27], q[38];
cx q[0], q[1];
rz(-0.2542710304260254) q[1];
cx q[0], q[1];
cx q[31], q[0];
rz(-0.2542710304260254) q[0];
cx q[31], q[0];
cx q[28], q[43];
rz(-0.2542710304260254) q[43];
cx q[28], q[43];
cx q[39], q[1];
rz(-0.2542710304260254) q[1];
cx q[39], q[1];
cx q[1], q[45];
rz(-0.2542710304260254) q[45];
cx q[1], q[45];
cx q[27], q[22];
rz(-0.2542710304260254) q[22];
cx q[27], q[22];
cx q[26], q[0];
rz(-0.2542710304260254) q[0];
cx q[26], q[0];
cx q[12], q[3];
rz(-0.2542710304260254) q[3];
cx q[12], q[3];
cx q[8], q[37];
rz(-0.2542710304260254) q[37];
cx q[8], q[37];
cx q[17], q[24];
rz(-0.2542710304260254) q[24];
cx q[17], q[24];
cx q[17], q[25];
rz(-0.2542710304260254) q[25];
cx q[17], q[25];
cx q[16], q[14];
rz(-0.2542710304260254) q[14];
cx q[16], q[14];
cx q[8], q[5];
rz(-0.2542710304260254) q[5];
cx q[8], q[5];
cx q[44], q[0];
rz(-0.2542710304260254) q[0];
cx q[44], q[0];
cx q[27], q[31];
rz(-0.2542710304260254) q[31];
cx q[27], q[31];
cx q[16], q[41];
rz(-0.2542710304260254) q[41];
cx q[16], q[41];
cx q[38], q[30];
rz(-0.2542710304260254) q[30];
cx q[38], q[30];
cx q[6], q[19];
rz(-0.2542710304260254) q[19];
cx q[6], q[19];
cx q[24], q[0];
rz(-0.2542710304260254) q[0];
cx q[24], q[0];
cx q[30], q[6];
rz(-0.2542710304260254) q[6];
cx q[30], q[6];
cx q[29], q[44];
rz(-0.2542710304260254) q[44];
cx q[29], q[44];
cx q[42], q[5];
rz(-0.2542710304260254) q[5];
cx q[42], q[5];
cx q[38], q[43];
rz(-0.2542710304260254) q[43];
cx q[38], q[43];
cx q[26], q[15];
rz(-0.2542710304260254) q[15];
cx q[26], q[15];
cx q[24], q[21];
rz(-0.2542710304260254) q[21];
cx q[24], q[21];
cx q[23], q[42];
rz(-0.2542710304260254) q[42];
cx q[23], q[42];
cx q[33], q[20];
rz(-0.2542710304260254) q[20];
cx q[33], q[20];
cx q[9], q[16];
rz(-0.2542710304260254) q[16];
cx q[9], q[16];
cx q[35], q[40];
rz(-0.2542710304260254) q[40];
cx q[35], q[40];
cx q[24], q[13];
rz(-0.2542710304260254) q[13];
cx q[24], q[13];
cx q[30], q[35];
rz(-0.2542710304260254) q[35];
cx q[30], q[35];
cx q[21], q[26];
rz(-0.2542710304260254) q[26];
cx q[21], q[26];
cx q[45], q[8];
rz(-0.2542710304260254) q[8];
cx q[45], q[8];
cx q[9], q[41];
rz(-0.2542710304260254) q[41];
cx q[9], q[41];
cx q[2], q[32];
rz(-0.2542710304260254) q[32];
cx q[2], q[32];
cx q[27], q[8];
rz(-0.2542710304260254) q[8];
cx q[27], q[8];
cx q[6], q[44];
rz(-0.2542710304260254) q[44];
cx q[6], q[44];
cx q[27], q[21];
rz(-0.2542710304260254) q[21];
cx q[27], q[21];
cx q[14], q[37];
rz(-0.2542710304260254) q[37];
cx q[14], q[37];
cx q[17], q[19];
rz(-0.2542710304260254) q[19];
cx q[17], q[19];
cx q[46], q[17];
rz(-0.2542710304260254) q[17];
cx q[46], q[17];
cx q[21], q[25];
rz(-0.2542710304260254) q[25];
cx q[21], q[25];
cx q[11], q[1];
rz(-0.2542710304260254) q[1];
cx q[11], q[1];
cx q[22], q[29];
rz(-0.2542710304260254) q[29];
cx q[22], q[29];
cx q[30], q[18];
rz(-0.2542710304260254) q[18];
cx q[30], q[18];
cx q[7], q[1];
rz(-0.2542710304260254) q[1];
cx q[7], q[1];
cx q[35], q[20];
rz(-0.2542710304260254) q[20];
cx q[35], q[20];
cx q[6], q[45];
rz(-0.2542710304260254) q[45];
cx q[6], q[45];
cx q[36], q[45];
rz(-0.2542710304260254) q[45];
cx q[36], q[45];
cx q[3], q[8];
rz(-0.2542710304260254) q[8];
cx q[3], q[8];
cx q[2], q[12];
rz(-0.2542710304260254) q[12];
cx q[2], q[12];
cx q[13], q[44];
rz(-0.2542710304260254) q[44];
cx q[13], q[44];
cx q[12], q[39];
rz(-0.2542710304260254) q[39];
cx q[12], q[39];
cx q[2], q[40];
rz(-0.2542710304260254) q[40];
cx q[2], q[40];
cx q[30], q[10];
rz(-0.2542710304260254) q[10];
cx q[30], q[10];
cx q[33], q[43];
rz(-0.2542710304260254) q[43];
cx q[33], q[43];
cx q[15], q[5];
rz(-0.2542710304260254) q[5];
cx q[15], q[5];
cx q[24], q[31];
rz(-0.2542710304260254) q[31];
cx q[24], q[31];
cx q[19], q[45];
rz(-0.2542710304260254) q[45];
cx q[19], q[45];
cx q[4], q[15];
rz(-0.2542710304260254) q[15];
cx q[4], q[15];
cx q[38], q[26];
rz(-0.2542710304260254) q[26];
cx q[38], q[26];
cx q[3], q[25];
rz(-0.2542710304260254) q[25];
cx q[3], q[25];
cx q[11], q[10];
rz(-0.2542710304260254) q[10];
cx q[11], q[10];
cx q[46], q[6];
rz(-0.2542710304260254) q[6];
cx q[46], q[6];
cx q[17], q[11];
rz(-0.2542710304260254) q[11];
cx q[17], q[11];
cx q[32], q[29];
rz(-0.2542710304260254) q[29];
cx q[32], q[29];
cx q[14], q[31];
rz(-0.2542710304260254) q[31];
cx q[14], q[31];
cx q[35], q[42];
rz(-0.2542710304260254) q[42];
cx q[35], q[42];
cx q[32], q[0];
rz(-0.2542710304260254) q[0];
cx q[32], q[0];
cx q[11], q[39];
rz(-0.2542710304260254) q[39];
cx q[11], q[39];
cx q[6], q[20];
rz(-0.2542710304260254) q[20];
cx q[6], q[20];
cx q[25], q[42];
rz(-0.2542710304260254) q[42];
cx q[25], q[42];
cx q[8], q[18];
rz(-0.2542710304260254) q[18];
cx q[8], q[18];
cx q[3], q[4];
rz(-0.2542710304260254) q[4];
cx q[3], q[4];
cx q[29], q[13];
rz(-0.2542710304260254) q[13];
cx q[29], q[13];
cx q[11], q[35];
rz(-0.2542710304260254) q[35];
cx q[11], q[35];
cx q[35], q[19];
rz(-0.2542710304260254) q[19];
cx q[35], q[19];
cx q[7], q[19];
rz(-0.2542710304260254) q[19];
cx q[7], q[19];
cx q[2], q[24];
rz(-0.2542710304260254) q[24];
cx q[2], q[24];
cx q[44], q[32];
rz(-0.2542710304260254) q[32];
cx q[44], q[32];
cx q[24], q[11];
rz(-0.2542710304260254) q[11];
cx q[24], q[11];
cx q[28], q[8];
rz(-0.2542710304260254) q[8];
cx q[28], q[8];
cx q[9], q[14];
rz(-0.2542710304260254) q[14];
cx q[9], q[14];
cx q[46], q[4];
rz(-0.2542710304260254) q[4];
cx q[46], q[4];
cx q[29], q[15];
rz(-0.2542710304260254) q[15];
cx q[29], q[15];
cx q[6], q[39];
rz(-0.2542710304260254) q[39];
cx q[6], q[39];
cx q[19], q[28];
rz(-0.2542710304260254) q[28];
cx q[19], q[28];
cx q[15], q[6];
rz(-0.2542710304260254) q[6];
cx q[15], q[6];
cx q[14], q[8];
rz(-0.2542710304260254) q[8];
cx q[14], q[8];
cx q[7], q[41];
rz(-0.2542710304260254) q[41];
cx q[7], q[41];
cx q[9], q[28];
rz(-0.2542710304260254) q[28];
cx q[9], q[28];
cx q[29], q[46];
rz(-0.2542710304260254) q[46];
cx q[29], q[46];
cx q[40], q[23];
rz(-0.2542710304260254) q[23];
cx q[40], q[23];
cx q[34], q[19];
rz(-0.2542710304260254) q[19];
cx q[34], q[19];
cx q[6], q[13];
rz(-0.2542710304260254) q[13];
cx q[6], q[13];
cx q[33], q[27];
rz(-0.2542710304260254) q[27];
cx q[33], q[27];
cx q[36], q[32];
rz(-0.2542710304260254) q[32];
cx q[36], q[32];
cx q[30], q[13];
rz(-0.2542710304260254) q[13];
cx q[30], q[13];
cx q[42], q[24];
rz(-0.2542710304260254) q[24];
cx q[42], q[24];
cx q[7], q[44];
rz(-0.2542710304260254) q[44];
cx q[7], q[44];
cx q[17], q[34];
rz(-0.2542710304260254) q[34];
cx q[17], q[34];
cx q[14], q[2];
rz(-0.2542710304260254) q[2];
cx q[14], q[2];
cx q[20], q[31];
rz(-0.2542710304260254) q[31];
cx q[20], q[31];
cx q[2], q[41];
rz(-0.2542710304260254) q[41];
cx q[2], q[41];
cx q[34], q[25];
rz(-0.2542710304260254) q[25];
cx q[34], q[25];
cx q[6], q[31];
rz(-0.2542710304260254) q[31];
cx q[6], q[31];
cx q[12], q[4];
rz(-0.2542710304260254) q[4];
cx q[12], q[4];
cx q[17], q[13];
rz(-0.2542710304260254) q[13];
cx q[17], q[13];
cx q[44], q[45];
rz(-0.2542710304260254) q[45];
cx q[44], q[45];
cx q[6], q[29];
rz(-0.2542710304260254) q[29];
cx q[6], q[29];
cx q[7], q[4];
rz(-0.2542710304260254) q[4];
cx q[7], q[4];
cx q[33], q[46];
rz(-0.2542710304260254) q[46];
cx q[33], q[46];
cx q[3], q[14];
rz(-0.2542710304260254) q[14];
cx q[3], q[14];
cx q[19], q[15];
rz(-0.2542710304260254) q[15];
cx q[19], q[15];
cx q[33], q[45];
rz(-0.2542710304260254) q[45];
cx q[33], q[45];
cx q[34], q[35];
rz(-0.2542710304260254) q[35];
cx q[34], q[35];
cx q[32], q[10];
rz(-0.2542710304260254) q[10];
cx q[32], q[10];
cx q[16], q[28];
rz(-0.2542710304260254) q[28];
cx q[16], q[28];
cx q[10], q[8];
rz(-0.2542710304260254) q[8];
cx q[10], q[8];
cx q[31], q[37];
rz(-0.2542710304260254) q[37];
cx q[31], q[37];
cx q[29], q[36];
rz(-0.2542710304260254) q[36];
cx q[29], q[36];
cx q[29], q[18];
rz(-0.2542710304260254) q[18];
cx q[29], q[18];
cx q[7], q[42];
rz(-0.2542710304260254) q[42];
cx q[7], q[42];
cx q[28], q[21];
rz(-0.2542710304260254) q[21];
cx q[28], q[21];
cx q[18], q[19];
rz(-0.2542710304260254) q[19];
cx q[18], q[19];
cx q[35], q[44];
rz(-0.2542710304260254) q[44];
cx q[35], q[44];
cx q[10], q[24];
rz(-0.2542710304260254) q[24];
cx q[10], q[24];
cx q[35], q[45];
rz(-0.2542710304260254) q[45];
cx q[35], q[45];
cx q[13], q[4];
rz(-0.2542710304260254) q[4];
cx q[13], q[4];
cx q[0], q[36];
rz(-0.2542710304260254) q[36];
cx q[0], q[36];
cx q[2], q[28];
rz(-0.2542710304260254) q[28];
cx q[2], q[28];
cx q[27], q[46];
rz(-0.2542710304260254) q[46];
cx q[27], q[46];
cx q[44], q[19];
rz(-0.2542710304260254) q[19];
cx q[44], q[19];
cx q[42], q[28];
rz(-0.2542710304260254) q[28];
cx q[42], q[28];
cx q[31], q[10];
rz(-0.2542710304260254) q[10];
cx q[31], q[10];
cx q[16], q[10];
rz(-0.2542710304260254) q[10];
cx q[16], q[10];
cx q[21], q[1];
rz(-0.2542710304260254) q[1];
cx q[21], q[1];
cx q[1], q[43];
rz(-0.2542710304260254) q[43];
cx q[1], q[43];
cx q[15], q[31];
rz(-0.2542710304260254) q[31];
cx q[15], q[31];
cx q[28], q[26];
rz(-0.2542710304260254) q[26];
cx q[28], q[26];
cx q[18], q[34];
rz(-0.2542710304260254) q[34];
cx q[18], q[34];
cx q[17], q[26];
rz(-0.2542710304260254) q[26];
cx q[17], q[26];
cx q[16], q[45];
rz(-0.2542710304260254) q[45];
cx q[16], q[45];
cx q[41], q[43];
rz(-0.2542710304260254) q[43];
cx q[41], q[43];
cx q[12], q[37];
rz(-0.2542710304260254) q[37];
cx q[12], q[37];
cx q[31], q[30];
rz(-0.2542710304260254) q[30];
cx q[31], q[30];
cx q[0], q[9];
rz(-0.2542710304260254) q[9];
cx q[0], q[9];
cx q[45], q[38];
rz(-0.2542710304260254) q[38];
cx q[45], q[38];
cx q[2], q[37];
rz(-0.2542710304260254) q[37];
cx q[2], q[37];
cx q[33], q[1];
rz(-0.2542710304260254) q[1];
cx q[33], q[1];
cx q[25], q[45];
rz(-0.2542710304260254) q[45];
cx q[25], q[45];
cx q[27], q[10];
rz(-0.2542710304260254) q[10];
cx q[27], q[10];
cx q[0], q[45];
rz(-0.2542710304260254) q[45];
cx q[0], q[45];
cx q[20], q[8];
rz(-0.2542710304260254) q[8];
cx q[20], q[8];
cx q[2], q[27];
rz(-0.2542710304260254) q[27];
cx q[2], q[27];
cx q[38], q[20];
rz(-0.2542710304260254) q[20];
cx q[38], q[20];
cx q[16], q[42];
rz(-0.2542710304260254) q[42];
cx q[16], q[42];
cx q[10], q[5];
rz(-0.2542710304260254) q[5];
cx q[10], q[5];
cx q[1], q[19];
rz(-0.2542710304260254) q[19];
cx q[1], q[19];
cx q[39], q[15];
rz(-0.2542710304260254) q[15];
cx q[39], q[15];
cx q[11], q[31];
rz(-0.2542710304260254) q[31];
cx q[11], q[31];
cx q[6], q[2];
rz(-0.2542710304260254) q[2];
cx q[6], q[2];
cx q[45], q[10];
rz(-0.2542710304260254) q[10];
cx q[45], q[10];
cx q[38], q[40];
rz(-0.2542710304260254) q[40];
cx q[38], q[40];
cx q[7], q[38];
rz(-0.2542710304260254) q[38];
cx q[7], q[38];
cx q[2], q[1];
rz(-0.2542710304260254) q[1];
cx q[2], q[1];
cx q[29], q[3];
rz(-0.2542710304260254) q[3];
cx q[29], q[3];
cx q[7], q[8];
rz(-0.2542710304260254) q[8];
cx q[7], q[8];
cx q[14], q[41];
rz(-0.2542710304260254) q[41];
cx q[14], q[41];
cx q[34], q[0];
rz(-0.2542710304260254) q[0];
cx q[34], q[0];
cx q[40], q[20];
rz(-0.2542710304260254) q[20];
cx q[40], q[20];
cx q[32], q[28];
rz(-0.2542710304260254) q[28];
cx q[32], q[28];
cx q[42], q[19];
rz(-0.2542710304260254) q[19];
cx q[42], q[19];
cx q[43], q[6];
rz(-0.2542710304260254) q[6];
cx q[43], q[6];
cx q[35], q[6];
rz(-0.2542710304260254) q[6];
cx q[35], q[6];
cx q[43], q[29];
rz(-0.2542710304260254) q[29];
cx q[43], q[29];
cx q[18], q[9];
rz(-0.2542710304260254) q[9];
cx q[18], q[9];
cx q[27], q[7];
rz(-0.2542710304260254) q[7];
cx q[27], q[7];
cx q[40], q[37];
rz(-0.2542710304260254) q[37];
cx q[40], q[37];
cx q[15], q[23];
rz(-0.2542710304260254) q[23];
cx q[15], q[23];
cx q[5], q[22];
rz(-0.2542710304260254) q[22];
cx q[5], q[22];
cx q[8], q[15];
rz(-0.2542710304260254) q[15];
cx q[8], q[15];
cx q[29], q[14];
rz(-0.2542710304260254) q[14];
cx q[29], q[14];
cx q[44], q[36];
rz(-0.2542710304260254) q[36];
cx q[44], q[36];
cx q[23], q[10];
rz(-0.2542710304260254) q[10];
cx q[23], q[10];
cx q[30], q[15];
rz(-0.2542710304260254) q[15];
cx q[30], q[15];
cx q[9], q[38];
rz(-0.2542710304260254) q[38];
cx q[9], q[38];
cx q[23], q[46];
rz(-0.2542710304260254) q[46];
cx q[23], q[46];
cx q[27], q[15];
rz(-0.2542710304260254) q[15];
cx q[27], q[15];
cx q[5], q[41];
rz(-0.2542710304260254) q[41];
cx q[5], q[41];
cx q[29], q[5];
rz(-0.2542710304260254) q[5];
cx q[29], q[5];
cx q[9], q[33];
rz(-0.2542710304260254) q[33];
cx q[9], q[33];
cx q[36], q[12];
rz(-0.2542710304260254) q[12];
cx q[36], q[12];
cx q[33], q[29];
rz(-0.2542710304260254) q[29];
cx q[33], q[29];
cx q[33], q[36];
rz(-0.2542710304260254) q[36];
cx q[33], q[36];
cx q[4], q[21];
rz(-0.2542710304260254) q[21];
cx q[4], q[21];
cx q[21], q[31];
rz(-0.2542710304260254) q[31];
cx q[21], q[31];
cx q[10], q[2];
rz(-0.2542710304260254) q[2];
cx q[10], q[2];
cx q[10], q[14];
rz(-0.2542710304260254) q[14];
cx q[10], q[14];
cx q[27], q[23];
rz(-0.2542710304260254) q[23];
cx q[27], q[23];
cx q[22], q[39];
rz(-0.2542710304260254) q[39];
cx q[22], q[39];
cx q[7], q[20];
rz(-0.2542710304260254) q[20];
cx q[7], q[20];
cx q[43], q[18];
rz(-0.2542710304260254) q[18];
cx q[43], q[18];
cx q[28], q[37];
rz(-0.2542710304260254) q[37];
cx q[28], q[37];
cx q[11], q[25];
rz(-0.2542710304260254) q[25];
cx q[11], q[25];
cx q[2], q[16];
rz(-0.2542710304260254) q[16];
cx q[2], q[16];
cx q[18], q[37];
rz(-0.2542710304260254) q[37];
cx q[18], q[37];
cx q[0], q[18];
rz(-0.2542710304260254) q[18];
cx q[0], q[18];
cx q[35], q[28];
rz(-0.2542710304260254) q[28];
cx q[35], q[28];
cx q[12], q[25];
rz(-0.2542710304260254) q[25];
cx q[12], q[25];
cx q[11], q[32];
rz(-0.2542710304260254) q[32];
cx q[11], q[32];
cx q[33], q[24];
rz(-0.2542710304260254) q[24];
cx q[33], q[24];
cx q[46], q[25];
rz(-0.2542710304260254) q[25];
cx q[46], q[25];
cx q[21], q[16];
rz(-0.2542710304260254) q[16];
cx q[21], q[16];
cx q[35], q[17];
rz(-0.2542710304260254) q[17];
cx q[35], q[17];
cx q[11], q[45];
rz(-0.2542710304260254) q[45];
cx q[11], q[45];
cx q[46], q[20];
rz(-0.2542710304260254) q[20];
cx q[46], q[20];
cx q[30], q[17];
rz(-0.2542710304260254) q[17];
cx q[30], q[17];
cx q[8], q[2];
rz(-0.2542710304260254) q[2];
cx q[8], q[2];
cx q[6], q[1];
rz(-0.2542710304260254) q[1];
cx q[6], q[1];
cx q[2], q[11];
rz(-0.2542710304260254) q[11];
cx q[2], q[11];
cx q[9], q[27];
rz(-0.2542710304260254) q[27];
cx q[9], q[27];
cx q[42], q[39];
rz(-0.2542710304260254) q[39];
cx q[42], q[39];
cx q[1], q[36];
rz(-0.2542710304260254) q[36];
cx q[1], q[36];
cx q[23], q[18];
rz(-0.2542710304260254) q[18];
cx q[23], q[18];
cx q[43], q[19];
rz(-0.2542710304260254) q[19];
cx q[43], q[19];
cx q[14], q[0];
rz(-0.2542710304260254) q[0];
cx q[14], q[0];
cx q[0], q[5];
rz(-0.2542710304260254) q[5];
cx q[0], q[5];
cx q[33], q[40];
rz(-0.2542710304260254) q[40];
cx q[33], q[40];
cx q[45], q[22];
rz(-0.2542710304260254) q[22];
cx q[45], q[22];
cx q[24], q[35];
rz(-0.2542710304260254) q[35];
cx q[24], q[35];
cx q[22], q[8];
rz(-0.2542710304260254) q[8];
cx q[22], q[8];
cx q[44], q[37];
rz(-0.2542710304260254) q[37];
cx q[44], q[37];
cx q[18], q[35];
rz(-0.2542710304260254) q[35];
cx q[18], q[35];
cx q[1], q[18];
rz(-0.2542710304260254) q[18];
cx q[1], q[18];
cx q[33], q[19];
rz(-0.2542710304260254) q[19];
cx q[33], q[19];
cx q[26], q[11];
rz(-0.2542710304260254) q[11];
cx q[26], q[11];
cx q[1], q[3];
rz(-0.2542710304260254) q[3];
cx q[1], q[3];
cx q[16], q[46];
rz(-0.2542710304260254) q[46];
cx q[16], q[46];
cx q[3], q[34];
rz(-0.2542710304260254) q[34];
cx q[3], q[34];
cx q[44], q[25];
rz(-0.2542710304260254) q[25];
cx q[44], q[25];
cx q[2], q[13];
rz(-0.2542710304260254) q[13];
cx q[2], q[13];
cx q[25], q[33];
rz(-0.2542710304260254) q[33];
cx q[25], q[33];
cx q[27], q[30];
rz(-0.2542710304260254) q[30];
cx q[27], q[30];
cx q[22], q[13];
rz(-0.2542710304260254) q[13];
cx q[22], q[13];
cx q[42], q[33];
rz(-0.2542710304260254) q[33];
cx q[42], q[33];
cx q[34], q[1];
rz(-0.2542710304260254) q[1];
cx q[34], q[1];
cx q[41], q[12];
rz(-0.2542710304260254) q[12];
cx q[41], q[12];
cx q[5], q[46];
rz(-0.2542710304260254) q[46];
cx q[5], q[46];
cx q[37], q[11];
rz(-0.2542710304260254) q[11];
cx q[37], q[11];
cx q[12], q[38];
rz(-0.2542710304260254) q[38];
cx q[12], q[38];
cx q[23], q[33];
rz(-0.2542710304260254) q[33];
cx q[23], q[33];
cx q[19], q[27];
rz(-0.2542710304260254) q[27];
cx q[19], q[27];
cx q[4], q[39];
rz(-0.2542710304260254) q[39];
cx q[4], q[39];
cx q[15], q[35];
rz(-0.2542710304260254) q[35];
cx q[15], q[35];
cx q[5], q[11];
rz(-0.2542710304260254) q[11];
cx q[5], q[11];
cx q[10], q[17];
rz(-0.2542710304260254) q[17];
cx q[10], q[17];
cx q[22], q[36];
rz(-0.2542710304260254) q[36];
cx q[22], q[36];
cx q[34], q[11];
rz(-0.2542710304260254) q[11];
cx q[34], q[11];
cx q[13], q[10];
rz(-0.2542710304260254) q[10];
cx q[13], q[10];
cx q[34], q[6];
rz(-0.2542710304260254) q[6];
cx q[34], q[6];
cx q[37], q[7];
rz(-0.2542710304260254) q[7];
cx q[37], q[7];
cx q[7], q[28];
rz(-0.2542710304260254) q[28];
cx q[7], q[28];
cx q[36], q[5];
rz(-0.2542710304260254) q[5];
cx q[36], q[5];
cx q[12], q[11];
rz(-0.2542710304260254) q[11];
cx q[12], q[11];
cx q[41], q[32];
rz(-0.2542710304260254) q[32];
cx q[41], q[32];
cx q[3], q[37];
rz(-0.2542710304260254) q[37];
cx q[3], q[37];
cx q[45], q[12];
rz(-0.2542710304260254) q[12];
cx q[45], q[12];
cx q[23], q[37];
rz(-0.2542710304260254) q[37];
cx q[23], q[37];
cx q[13], q[34];
rz(-0.2542710304260254) q[34];
cx q[13], q[34];
cx q[1], q[35];
rz(-0.2542710304260254) q[35];
cx q[1], q[35];
cx q[16], q[18];
rz(-0.2542710304260254) q[18];
cx q[16], q[18];
cx q[25], q[7];
rz(-0.2542710304260254) q[7];
cx q[25], q[7];
cx q[20], q[17];
rz(-0.2542710304260254) q[17];
cx q[20], q[17];
cx q[17], q[31];
rz(-0.2542710304260254) q[31];
cx q[17], q[31];
cx q[45], q[23];
rz(-0.2542710304260254) q[23];
cx q[45], q[23];
cx q[29], q[2];
rz(-0.2542710304260254) q[2];
cx q[29], q[2];
cx q[30], q[11];
rz(-0.2542710304260254) q[11];
cx q[30], q[11];
cx q[22], q[26];
rz(-0.2542710304260254) q[26];
cx q[22], q[26];
cx q[24], q[29];
rz(-0.2542710304260254) q[29];
cx q[24], q[29];
cx q[32], q[5];
rz(-0.2542710304260254) q[5];
cx q[32], q[5];
cx q[40], q[0];
rz(-0.2542710304260254) q[0];
cx q[40], q[0];
cx q[30], q[22];
rz(-0.2542710304260254) q[22];
cx q[30], q[22];
cx q[9], q[39];
rz(-0.2542710304260254) q[39];
cx q[9], q[39];
cx q[43], q[0];
rz(-0.2542710304260254) q[0];
cx q[43], q[0];
cx q[46], q[45];
rz(-0.2542710304260254) q[45];
cx q[46], q[45];
cx q[40], q[24];
rz(-0.2542710304260254) q[24];
cx q[40], q[24];
cx q[14], q[40];
rz(-0.2542710304260254) q[40];
cx q[14], q[40];
cx q[13], q[45];
rz(-0.2542710304260254) q[45];
cx q[13], q[45];
cx q[30], q[7];
rz(-0.2542710304260254) q[7];
cx q[30], q[7];
cx q[46], q[0];
rz(-0.2542710304260254) q[0];
cx q[46], q[0];
cx q[13], q[9];
rz(-0.2542710304260254) q[9];
cx q[13], q[9];
cx q[26], q[43];
rz(-0.2542710304260254) q[43];
cx q[26], q[43];
cx q[32], q[22];
rz(-0.2542710304260254) q[22];
cx q[32], q[22];
cx q[22], q[43];
rz(-0.2542710304260254) q[43];
cx q[22], q[43];
cx q[46], q[3];
rz(-0.2542710304260254) q[3];
cx q[46], q[3];
cx q[24], q[38];
rz(-0.2542710304260254) q[38];
cx q[24], q[38];
cx q[12], q[9];
rz(-0.2542710304260254) q[9];
cx q[12], q[9];
cx q[23], q[0];
rz(-0.2542710304260254) q[0];
cx q[23], q[0];
cx q[22], q[15];
rz(-0.2542710304260254) q[15];
cx q[22], q[15];
cx q[33], q[12];
rz(-0.2542710304260254) q[12];
cx q[33], q[12];
cx q[44], q[24];
rz(-0.2542710304260254) q[24];
cx q[44], q[24];
cx q[4], q[20];
rz(-0.2542710304260254) q[20];
cx q[4], q[20];
cx q[35], q[4];
rz(-0.2542710304260254) q[4];
cx q[35], q[4];
cx q[12], q[44];
rz(-0.2542710304260254) q[44];
cx q[12], q[44];
cx q[40], q[44];
rz(-0.2542710304260254) q[44];
cx q[40], q[44];
cx q[17], q[32];
rz(-0.2542710304260254) q[32];
cx q[17], q[32];
cx q[38], q[37];
rz(-0.2542710304260254) q[37];
cx q[38], q[37];
cx q[5], q[45];
rz(-0.2542710304260254) q[45];
cx q[5], q[45];
cx q[33], q[37];
rz(-0.2542710304260254) q[37];
cx q[33], q[37];
cx q[12], q[10];
rz(-0.2542710304260254) q[10];
cx q[12], q[10];
cx q[7], q[31];
rz(-0.2542710304260254) q[31];
cx q[7], q[31];
cx q[8], q[40];
rz(-0.2542710304260254) q[40];
cx q[8], q[40];
cx q[11], q[33];
rz(-0.2542710304260254) q[33];
cx q[11], q[33];
cx q[30], q[37];
rz(-0.2542710304260254) q[37];
cx q[30], q[37];
cx q[20], q[45];
rz(-0.2542710304260254) q[45];
cx q[20], q[45];
cx q[0], q[37];
rz(-0.2542710304260254) q[37];
cx q[0], q[37];
cx q[37], q[5];
rz(-0.2542710304260254) q[5];
cx q[37], q[5];
cx q[23], q[44];
rz(-0.2542710304260254) q[44];
cx q[23], q[44];
cx q[41], q[33];
rz(-0.2542710304260254) q[33];
cx q[41], q[33];
cx q[18], q[27];
rz(-0.2542710304260254) q[27];
cx q[18], q[27];
cx q[16], q[19];
rz(-0.2542710304260254) q[19];
cx q[16], q[19];
cx q[0], q[22];
rz(-0.2542710304260254) q[22];
cx q[0], q[22];
cx q[12], q[34];
rz(-0.2542710304260254) q[34];
cx q[12], q[34];
cx q[39], q[46];
rz(-0.2542710304260254) q[46];
cx q[39], q[46];
cx q[29], q[28];
rz(-0.2542710304260254) q[28];
cx q[29], q[28];
cx q[15], q[0];
rz(-0.2542710304260254) q[0];
cx q[15], q[0];
cx q[15], q[14];
rz(-0.2542710304260254) q[14];
cx q[15], q[14];
cx q[21], q[2];
rz(-0.2542710304260254) q[2];
cx q[21], q[2];
cx q[6], q[10];
rz(-0.2542710304260254) q[10];
cx q[6], q[10];
cx q[20], q[23];
rz(-0.2542710304260254) q[23];
cx q[20], q[23];
cx q[39], q[35];
rz(-0.2542710304260254) q[35];
cx q[39], q[35];
cx q[32], q[9];
rz(-0.2542710304260254) q[9];
cx q[32], q[9];
cx q[39], q[32];
rz(-0.2542710304260254) q[32];
cx q[39], q[32];
cx q[2], q[43];
rz(-0.2542710304260254) q[43];
cx q[2], q[43];
cx q[44], q[2];
rz(-0.2542710304260254) q[2];
cx q[44], q[2];
cx q[34], q[26];
rz(-0.2542710304260254) q[26];
cx q[34], q[26];
cx q[13], q[41];
rz(-0.2542710304260254) q[41];
cx q[13], q[41];
cx q[10], q[21];
rz(-0.2542710304260254) q[21];
cx q[10], q[21];
cx q[27], q[40];
rz(-0.2542710304260254) q[40];
cx q[27], q[40];
cx q[37], q[20];
rz(-0.2542710304260254) q[20];
cx q[37], q[20];
cx q[1], q[15];
rz(-0.2542710304260254) q[15];
cx q[1], q[15];
cx q[11], q[13];
rz(-0.2542710304260254) q[13];
cx q[11], q[13];
cx q[35], q[0];
rz(-0.2542710304260254) q[0];
cx q[35], q[0];
cx q[1], q[12];
rz(-0.2542710304260254) q[12];
cx q[1], q[12];
cx q[29], q[0];
rz(-0.2542710304260254) q[0];
cx q[29], q[0];
cx q[26], q[19];
rz(-0.2542710304260254) q[19];
cx q[26], q[19];
cx q[24], q[26];
rz(-0.2542710304260254) q[26];
cx q[24], q[26];
cx q[42], q[2];
rz(-0.2542710304260254) q[2];
cx q[42], q[2];
cx q[36], q[9];
rz(-0.2542710304260254) q[9];
cx q[36], q[9];
cx q[25], q[18];
rz(-0.2542710304260254) q[18];
cx q[25], q[18];
cx q[44], q[30];
rz(-0.2542710304260254) q[30];
cx q[44], q[30];
cx q[13], q[37];
rz(-0.2542710304260254) q[37];
cx q[13], q[37];
cx q[40], q[16];
rz(-0.2542710304260254) q[16];
cx q[40], q[16];
cx q[18], q[4];
rz(-0.2542710304260254) q[4];
cx q[18], q[4];
cx q[38], q[10];
rz(-0.2542710304260254) q[10];
cx q[38], q[10];
cx q[30], q[2];
rz(-0.2542710304260254) q[2];
cx q[30], q[2];
cx q[26], q[3];
rz(-0.2542710304260254) q[3];
cx q[26], q[3];
cx q[28], q[3];
rz(-0.2542710304260254) q[3];
cx q[28], q[3];
cx q[9], q[25];
rz(-0.2542710304260254) q[25];
cx q[9], q[25];
cx q[8], q[39];
rz(-0.2542710304260254) q[39];
cx q[8], q[39];
cx q[2], q[46];
rz(-0.2542710304260254) q[46];
cx q[2], q[46];
cx q[34], q[5];
rz(-0.2542710304260254) q[5];
cx q[34], q[5];
cx q[21], q[23];
rz(-0.2542710304260254) q[23];
cx q[21], q[23];
cx q[10], q[41];
rz(-0.2542710304260254) q[41];
cx q[10], q[41];
cx q[0], q[3];
rz(-0.2542710304260254) q[3];
cx q[0], q[3];
cx q[20], q[34];
rz(-0.2542710304260254) q[34];
cx q[20], q[34];
cx q[7], q[29];
rz(-0.2542710304260254) q[29];
cx q[7], q[29];
cx q[46], q[43];
rz(-0.2542710304260254) q[43];
cx q[46], q[43];
cx q[44], q[43];
rz(-0.2542710304260254) q[43];
cx q[44], q[43];
cx q[46], q[18];
rz(-0.2542710304260254) q[18];
cx q[46], q[18];
cx q[21], q[12];
rz(-0.2542710304260254) q[12];
cx q[21], q[12];
cx q[44], q[21];
rz(-0.2542710304260254) q[21];
cx q[44], q[21];
cx q[40], q[25];
rz(-0.2542710304260254) q[25];
cx q[40], q[25];
cx q[8], q[24];
rz(-0.2542710304260254) q[24];
cx q[8], q[24];
cx q[4], q[33];
rz(-0.2542710304260254) q[33];
cx q[4], q[33];
cx q[28], q[45];
rz(-0.2542710304260254) q[45];
cx q[28], q[45];
cx q[3], q[11];
rz(-0.2542710304260254) q[11];
cx q[3], q[11];
cx q[43], q[15];
rz(-0.2542710304260254) q[15];
cx q[43], q[15];
cx q[11], q[20];
rz(-0.2542710304260254) q[20];
cx q[11], q[20];
cx q[35], q[33];
rz(-0.2542710304260254) q[33];
cx q[35], q[33];
cx q[25], q[1];
rz(-0.2542710304260254) q[1];
cx q[25], q[1];
cx q[22], q[10];
rz(-0.2542710304260254) q[10];
cx q[22], q[10];
cx q[16], q[32];
rz(-0.2542710304260254) q[32];
cx q[16], q[32];
cx q[44], q[22];
rz(-0.2542710304260254) q[22];
cx q[44], q[22];
cx q[30], q[39];
rz(-0.2542710304260254) q[39];
cx q[30], q[39];
cx q[28], q[10];
rz(-0.2542710304260254) q[10];
cx q[28], q[10];
cx q[19], q[39];
rz(-0.2542710304260254) q[39];
cx q[19], q[39];
cx q[35], q[7];
rz(-0.2542710304260254) q[7];
cx q[35], q[7];
cx q[4], q[30];
rz(-0.2542710304260254) q[30];
cx q[4], q[30];
cx q[25], q[8];
rz(-0.2542710304260254) q[8];
cx q[25], q[8];
cx q[34], q[32];
rz(-0.2542710304260254) q[32];
cx q[34], q[32];
rx(1.3564391136169434) q[0];
rx(1.3564391136169434) q[1];
rx(1.3564391136169434) q[2];
rx(1.3564391136169434) q[3];
rx(1.3564391136169434) q[4];
rx(1.3564391136169434) q[5];
rx(1.3564391136169434) q[6];
rx(1.3564391136169434) q[7];
rx(1.3564391136169434) q[8];
rx(1.3564391136169434) q[9];
rx(1.3564391136169434) q[10];
rx(1.3564391136169434) q[11];
rx(1.3564391136169434) q[12];
rx(1.3564391136169434) q[13];
rx(1.3564391136169434) q[14];
rx(1.3564391136169434) q[15];
rx(1.3564391136169434) q[16];
rx(1.3564391136169434) q[17];
rx(1.3564391136169434) q[18];
rx(1.3564391136169434) q[19];
rx(1.3564391136169434) q[20];
rx(1.3564391136169434) q[21];
rx(1.3564391136169434) q[22];
rx(1.3564391136169434) q[23];
rx(1.3564391136169434) q[24];
rx(1.3564391136169434) q[25];
rx(1.3564391136169434) q[26];
rx(1.3564391136169434) q[27];
rx(1.3564391136169434) q[28];
rx(1.3564391136169434) q[29];
rx(1.3564391136169434) q[30];
rx(1.3564391136169434) q[31];
rx(1.3564391136169434) q[32];
rx(1.3564391136169434) q[33];
rx(1.3564391136169434) q[34];
rx(1.3564391136169434) q[35];
rx(1.3564391136169434) q[36];
rx(1.3564391136169434) q[37];
rx(1.3564391136169434) q[38];
rx(1.3564391136169434) q[39];
rx(1.3564391136169434) q[40];
rx(1.3564391136169434) q[41];
rx(1.3564391136169434) q[42];
rx(1.3564391136169434) q[43];
rx(1.3564391136169434) q[44];
rx(1.3564391136169434) q[45];
rx(1.3564391136169434) q[46];