OPENQASM 2.0;
include "qelib1.inc";
qreg q[85];
creg c[85];
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
h q[47];
h q[48];
h q[49];
h q[50];
h q[51];
h q[52];
h q[53];
h q[54];
h q[55];
h q[56];
h q[57];
h q[58];
h q[59];
h q[60];
h q[61];
h q[62];
h q[63];
h q[64];
h q[65];
h q[66];
h q[67];
h q[68];
h q[69];
h q[70];
h q[71];
h q[72];
h q[73];
h q[74];
h q[75];
h q[76];
h q[77];
h q[78];
h q[79];
h q[80];
h q[81];
h q[82];
h q[83];
h q[84];
cx q[4], q[60];
rz(-1.6413257122039795) q[60];
cx q[4], q[60];
cx q[65], q[37];
rz(-1.6413257122039795) q[37];
cx q[65], q[37];
cx q[20], q[8];
rz(-1.6413257122039795) q[8];
cx q[20], q[8];
cx q[1], q[43];
rz(-1.6413257122039795) q[43];
cx q[1], q[43];
rx(1.9843093156814575) q[0];
rx(1.9843093156814575) q[1];
rx(1.9843093156814575) q[2];
rx(1.9843093156814575) q[3];
rx(1.9843093156814575) q[4];
rx(1.9843093156814575) q[5];
rx(1.9843093156814575) q[6];
rx(1.9843093156814575) q[7];
rx(1.9843093156814575) q[8];
rx(1.9843093156814575) q[9];
rx(1.9843093156814575) q[10];
rx(1.9843093156814575) q[11];
rx(1.9843093156814575) q[12];
rx(1.9843093156814575) q[13];
rx(1.9843093156814575) q[14];
rx(1.9843093156814575) q[15];
rx(1.9843093156814575) q[16];
rx(1.9843093156814575) q[17];
rx(1.9843093156814575) q[18];
rx(1.9843093156814575) q[19];
rx(1.9843093156814575) q[20];
rx(1.9843093156814575) q[21];
rx(1.9843093156814575) q[22];
rx(1.9843093156814575) q[23];
rx(1.9843093156814575) q[24];
rx(1.9843093156814575) q[25];
rx(1.9843093156814575) q[26];
rx(1.9843093156814575) q[27];
rx(1.9843093156814575) q[28];
rx(1.9843093156814575) q[29];
rx(1.9843093156814575) q[30];
rx(1.9843093156814575) q[31];
rx(1.9843093156814575) q[32];
rx(1.9843093156814575) q[33];
rx(1.9843093156814575) q[34];
rx(1.9843093156814575) q[35];
rx(1.9843093156814575) q[36];
rx(1.9843093156814575) q[37];
rx(1.9843093156814575) q[38];
rx(1.9843093156814575) q[39];
rx(1.9843093156814575) q[40];
rx(1.9843093156814575) q[41];
rx(1.9843093156814575) q[42];
rx(1.9843093156814575) q[43];
rx(1.9843093156814575) q[44];
rx(1.9843093156814575) q[45];
rx(1.9843093156814575) q[46];
rx(1.9843093156814575) q[47];
rx(1.9843093156814575) q[48];
rx(1.9843093156814575) q[49];
rx(1.9843093156814575) q[50];
rx(1.9843093156814575) q[51];
rx(1.9843093156814575) q[52];
rx(1.9843093156814575) q[53];
rx(1.9843093156814575) q[54];
rx(1.9843093156814575) q[55];
rx(1.9843093156814575) q[56];
rx(1.9843093156814575) q[57];
rx(1.9843093156814575) q[58];
rx(1.9843093156814575) q[59];
rx(1.9843093156814575) q[60];
rx(1.9843093156814575) q[61];
rx(1.9843093156814575) q[62];
rx(1.9843093156814575) q[63];
rx(1.9843093156814575) q[64];
rx(1.9843093156814575) q[65];
rx(1.9843093156814575) q[66];
rx(1.9843093156814575) q[67];
rx(1.9843093156814575) q[68];
rx(1.9843093156814575) q[69];
rx(1.9843093156814575) q[70];
rx(1.9843093156814575) q[71];
rx(1.9843093156814575) q[72];
rx(1.9843093156814575) q[73];
rx(1.9843093156814575) q[74];
rx(1.9843093156814575) q[75];
rx(1.9843093156814575) q[76];
rx(1.9843093156814575) q[77];
rx(1.9843093156814575) q[78];
rx(1.9843093156814575) q[79];
rx(1.9843093156814575) q[80];
rx(1.9843093156814575) q[81];
rx(1.9843093156814575) q[82];
rx(1.9843093156814575) q[83];
rx(1.9843093156814575) q[84];
