OPENQASM 2.0;
include "qelib1.inc";
qreg q[38];
creg c[38];
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
cx q[32], q[11];
rz(-0.5898352861404419) q[11];
cx q[32], q[11];
cx q[36], q[10];
rz(-0.5898352861404419) q[10];
cx q[36], q[10];
cx q[26], q[30];
rz(-0.5898352861404419) q[30];
cx q[26], q[30];
cx q[11], q[31];
rz(-0.5898352861404419) q[31];
cx q[11], q[31];
cx q[12], q[32];
rz(-0.5898352861404419) q[32];
cx q[12], q[32];
cx q[22], q[19];
rz(-0.5898352861404419) q[19];
cx q[22], q[19];
cx q[21], q[30];
rz(-0.5898352861404419) q[30];
cx q[21], q[30];
cx q[22], q[33];
rz(-0.5898352861404419) q[33];
cx q[22], q[33];
cx q[9], q[16];
rz(-0.5898352861404419) q[16];
cx q[9], q[16];
cx q[24], q[26];
rz(-0.5898352861404419) q[26];
cx q[24], q[26];
cx q[18], q[2];
rz(-0.5898352861404419) q[2];
cx q[18], q[2];
cx q[6], q[7];
rz(-0.5898352861404419) q[7];
cx q[6], q[7];
cx q[13], q[34];
rz(-0.5898352861404419) q[34];
cx q[13], q[34];
cx q[37], q[24];
rz(-0.5898352861404419) q[24];
cx q[37], q[24];
cx q[26], q[1];
rz(-0.5898352861404419) q[1];
cx q[26], q[1];
cx q[8], q[32];
rz(-0.5898352861404419) q[32];
cx q[8], q[32];
cx q[21], q[14];
rz(-0.5898352861404419) q[14];
cx q[21], q[14];
cx q[34], q[29];
rz(-0.5898352861404419) q[29];
cx q[34], q[29];
cx q[2], q[23];
rz(-0.5898352861404419) q[23];
cx q[2], q[23];
cx q[24], q[35];
rz(-0.5898352861404419) q[35];
cx q[24], q[35];
cx q[31], q[20];
rz(-0.5898352861404419) q[20];
cx q[31], q[20];
cx q[18], q[12];
rz(-0.5898352861404419) q[12];
cx q[18], q[12];
cx q[29], q[2];
rz(-0.5898352861404419) q[2];
cx q[29], q[2];
cx q[24], q[4];
rz(-0.5898352861404419) q[4];
cx q[24], q[4];
cx q[31], q[24];
rz(-0.5898352861404419) q[24];
cx q[31], q[24];
cx q[27], q[18];
rz(-0.5898352861404419) q[18];
cx q[27], q[18];
cx q[37], q[13];
rz(-0.5898352861404419) q[13];
cx q[37], q[13];
cx q[34], q[0];
rz(-0.5898352861404419) q[0];
cx q[34], q[0];
cx q[27], q[9];
rz(-0.5898352861404419) q[9];
cx q[27], q[9];
cx q[14], q[16];
rz(-0.5898352861404419) q[16];
cx q[14], q[16];
cx q[1], q[36];
rz(-0.5898352861404419) q[36];
cx q[1], q[36];
cx q[6], q[13];
rz(-0.5898352861404419) q[13];
cx q[6], q[13];
cx q[8], q[20];
rz(-0.5898352861404419) q[20];
cx q[8], q[20];
cx q[12], q[8];
rz(-0.5898352861404419) q[8];
cx q[12], q[8];
cx q[35], q[25];
rz(-0.5898352861404419) q[25];
cx q[35], q[25];
cx q[16], q[15];
rz(-0.5898352861404419) q[15];
cx q[16], q[15];
cx q[22], q[3];
rz(-0.5898352861404419) q[3];
cx q[22], q[3];
cx q[7], q[30];
rz(-0.5898352861404419) q[30];
cx q[7], q[30];
cx q[10], q[1];
rz(-0.5898352861404419) q[1];
cx q[10], q[1];
cx q[3], q[20];
rz(-0.5898352861404419) q[20];
cx q[3], q[20];
cx q[11], q[26];
rz(-0.5898352861404419) q[26];
cx q[11], q[26];
cx q[13], q[14];
rz(-0.5898352861404419) q[14];
cx q[13], q[14];
cx q[36], q[12];
rz(-0.5898352861404419) q[12];
cx q[36], q[12];
cx q[15], q[4];
rz(-0.5898352861404419) q[4];
cx q[15], q[4];
cx q[9], q[11];
rz(-0.5898352861404419) q[11];
cx q[9], q[11];
cx q[25], q[15];
rz(-0.5898352861404419) q[15];
cx q[25], q[15];
cx q[28], q[9];
rz(-0.5898352861404419) q[9];
cx q[28], q[9];
cx q[14], q[30];
rz(-0.5898352861404419) q[30];
cx q[14], q[30];
cx q[8], q[17];
rz(-0.5898352861404419) q[17];
cx q[8], q[17];
cx q[22], q[1];
rz(-0.5898352861404419) q[1];
cx q[22], q[1];
cx q[15], q[33];
rz(-0.5898352861404419) q[33];
cx q[15], q[33];
cx q[3], q[27];
rz(-0.5898352861404419) q[27];
cx q[3], q[27];
cx q[36], q[11];
rz(-0.5898352861404419) q[11];
cx q[36], q[11];
cx q[10], q[14];
rz(-0.5898352861404419) q[14];
cx q[10], q[14];
cx q[8], q[1];
rz(-0.5898352861404419) q[1];
cx q[8], q[1];
cx q[25], q[31];
rz(-0.5898352861404419) q[31];
cx q[25], q[31];
cx q[15], q[24];
rz(-0.5898352861404419) q[24];
cx q[15], q[24];
cx q[6], q[10];
rz(-0.5898352861404419) q[10];
cx q[6], q[10];
cx q[32], q[15];
rz(-0.5898352861404419) q[15];
cx q[32], q[15];
cx q[30], q[28];
rz(-0.5898352861404419) q[28];
cx q[30], q[28];
cx q[11], q[27];
rz(-0.5898352861404419) q[27];
cx q[11], q[27];
cx q[6], q[24];
rz(-0.5898352861404419) q[24];
cx q[6], q[24];
cx q[34], q[17];
rz(-0.5898352861404419) q[17];
cx q[34], q[17];
cx q[2], q[16];
rz(-0.5898352861404419) q[16];
cx q[2], q[16];
cx q[6], q[22];
rz(-0.5898352861404419) q[22];
cx q[6], q[22];
cx q[34], q[37];
rz(-0.5898352861404419) q[37];
cx q[34], q[37];
cx q[11], q[28];
rz(-0.5898352861404419) q[28];
cx q[11], q[28];
cx q[20], q[15];
rz(-0.5898352861404419) q[15];
cx q[20], q[15];
cx q[25], q[19];
rz(-0.5898352861404419) q[19];
cx q[25], q[19];
cx q[4], q[27];
rz(-0.5898352861404419) q[27];
cx q[4], q[27];
cx q[8], q[23];
rz(-0.5898352861404419) q[23];
cx q[8], q[23];
cx q[16], q[37];
rz(-0.5898352861404419) q[37];
cx q[16], q[37];
cx q[34], q[26];
rz(-0.5898352861404419) q[26];
cx q[34], q[26];
cx q[3], q[14];
rz(-0.5898352861404419) q[14];
cx q[3], q[14];
cx q[9], q[34];
rz(-0.5898352861404419) q[34];
cx q[9], q[34];
cx q[26], q[6];
rz(-0.5898352861404419) q[6];
cx q[26], q[6];
cx q[18], q[28];
rz(-0.5898352861404419) q[28];
cx q[18], q[28];
cx q[33], q[25];
rz(-0.5898352861404419) q[25];
cx q[33], q[25];
cx q[36], q[33];
rz(-0.5898352861404419) q[33];
cx q[36], q[33];
cx q[0], q[17];
rz(-0.5898352861404419) q[17];
cx q[0], q[17];
cx q[14], q[27];
rz(-0.5898352861404419) q[27];
cx q[14], q[27];
cx q[36], q[23];
rz(-0.5898352861404419) q[23];
cx q[36], q[23];
cx q[22], q[21];
rz(-0.5898352861404419) q[21];
cx q[22], q[21];
cx q[5], q[0];
rz(-0.5898352861404419) q[0];
cx q[5], q[0];
cx q[6], q[4];
rz(-0.5898352861404419) q[4];
cx q[6], q[4];
cx q[18], q[32];
rz(-0.5898352861404419) q[32];
cx q[18], q[32];
cx q[7], q[29];
rz(-0.5898352861404419) q[29];
cx q[7], q[29];
cx q[22], q[16];
rz(-0.5898352861404419) q[16];
cx q[22], q[16];
cx q[33], q[31];
rz(-0.5898352861404419) q[31];
cx q[33], q[31];
cx q[35], q[16];
rz(-0.5898352861404419) q[16];
cx q[35], q[16];
cx q[6], q[23];
rz(-0.5898352861404419) q[23];
cx q[6], q[23];
cx q[30], q[6];
rz(-0.5898352861404419) q[6];
cx q[30], q[6];
cx q[5], q[32];
rz(-0.5898352861404419) q[32];
cx q[5], q[32];
cx q[12], q[26];
rz(-0.5898352861404419) q[26];
cx q[12], q[26];
cx q[24], q[17];
rz(-0.5898352861404419) q[17];
cx q[24], q[17];
cx q[32], q[26];
rz(-0.5898352861404419) q[26];
cx q[32], q[26];
cx q[33], q[29];
rz(-0.5898352861404419) q[29];
cx q[33], q[29];
cx q[28], q[16];
rz(-0.5898352861404419) q[16];
cx q[28], q[16];
cx q[13], q[24];
rz(-0.5898352861404419) q[24];
cx q[13], q[24];
cx q[3], q[23];
rz(-0.5898352861404419) q[23];
cx q[3], q[23];
cx q[29], q[36];
rz(-0.5898352861404419) q[36];
cx q[29], q[36];
cx q[19], q[3];
rz(-0.5898352861404419) q[3];
cx q[19], q[3];
cx q[0], q[1];
rz(-0.5898352861404419) q[1];
cx q[0], q[1];
cx q[33], q[19];
rz(-0.5898352861404419) q[19];
cx q[33], q[19];
cx q[15], q[31];
rz(-0.5898352861404419) q[31];
cx q[15], q[31];
cx q[37], q[15];
rz(-0.5898352861404419) q[15];
cx q[37], q[15];
cx q[20], q[21];
rz(-0.5898352861404419) q[21];
cx q[20], q[21];
cx q[24], q[29];
rz(-0.5898352861404419) q[29];
cx q[24], q[29];
cx q[18], q[34];
rz(-0.5898352861404419) q[34];
cx q[18], q[34];
cx q[5], q[4];
rz(-0.5898352861404419) q[4];
cx q[5], q[4];
cx q[29], q[8];
rz(-0.5898352861404419) q[8];
cx q[29], q[8];
cx q[14], q[33];
rz(-0.5898352861404419) q[33];
cx q[14], q[33];
cx q[11], q[10];
rz(-0.5898352861404419) q[10];
cx q[11], q[10];
cx q[16], q[30];
rz(-0.5898352861404419) q[30];
cx q[16], q[30];
cx q[31], q[12];
rz(-0.5898352861404419) q[12];
cx q[31], q[12];
cx q[0], q[3];
rz(-0.5898352861404419) q[3];
cx q[0], q[3];
cx q[9], q[23];
rz(-0.5898352861404419) q[23];
cx q[9], q[23];
cx q[19], q[17];
rz(-0.5898352861404419) q[17];
cx q[19], q[17];
cx q[14], q[20];
rz(-0.5898352861404419) q[20];
cx q[14], q[20];
cx q[14], q[31];
rz(-0.5898352861404419) q[31];
cx q[14], q[31];
cx q[34], q[21];
rz(-0.5898352861404419) q[21];
cx q[34], q[21];
cx q[8], q[16];
rz(-0.5898352861404419) q[16];
cx q[8], q[16];
cx q[8], q[14];
rz(-0.5898352861404419) q[14];
cx q[8], q[14];
cx q[6], q[16];
rz(-0.5898352861404419) q[16];
cx q[6], q[16];
cx q[29], q[1];
rz(-0.5898352861404419) q[1];
cx q[29], q[1];
cx q[11], q[33];
rz(-0.5898352861404419) q[33];
cx q[11], q[33];
cx q[21], q[17];
rz(-0.5898352861404419) q[17];
cx q[21], q[17];
cx q[15], q[23];
rz(-0.5898352861404419) q[23];
cx q[15], q[23];
cx q[2], q[31];
rz(-0.5898352861404419) q[31];
cx q[2], q[31];
cx q[29], q[37];
rz(-0.5898352861404419) q[37];
cx q[29], q[37];
cx q[0], q[2];
rz(-0.5898352861404419) q[2];
cx q[0], q[2];
cx q[31], q[34];
rz(-0.5898352861404419) q[34];
cx q[31], q[34];
cx q[37], q[30];
rz(-0.5898352861404419) q[30];
cx q[37], q[30];
cx q[10], q[3];
rz(-0.5898352861404419) q[3];
cx q[10], q[3];
cx q[9], q[30];
rz(-0.5898352861404419) q[30];
cx q[9], q[30];
cx q[19], q[35];
rz(-0.5898352861404419) q[35];
cx q[19], q[35];
cx q[9], q[8];
rz(-0.5898352861404419) q[8];
cx q[9], q[8];
cx q[0], q[8];
rz(-0.5898352861404419) q[8];
cx q[0], q[8];
cx q[37], q[3];
rz(-0.5898352861404419) q[3];
cx q[37], q[3];
cx q[31], q[18];
rz(-0.5898352861404419) q[18];
cx q[31], q[18];
cx q[13], q[27];
rz(-0.5898352861404419) q[27];
cx q[13], q[27];
cx q[14], q[36];
rz(-0.5898352861404419) q[36];
cx q[14], q[36];
cx q[27], q[0];
rz(-0.5898352861404419) q[0];
cx q[27], q[0];
cx q[34], q[25];
rz(-0.5898352861404419) q[25];
cx q[34], q[25];
cx q[24], q[9];
rz(-0.5898352861404419) q[9];
cx q[24], q[9];
cx q[26], q[29];
rz(-0.5898352861404419) q[29];
cx q[26], q[29];
cx q[25], q[1];
rz(-0.5898352861404419) q[1];
cx q[25], q[1];
cx q[29], q[0];
rz(-0.5898352861404419) q[0];
cx q[29], q[0];
cx q[29], q[32];
rz(-0.5898352861404419) q[32];
cx q[29], q[32];
cx q[29], q[6];
rz(-0.5898352861404419) q[6];
cx q[29], q[6];
cx q[33], q[1];
rz(-0.5898352861404419) q[1];
cx q[33], q[1];
cx q[13], q[8];
rz(-0.5898352861404419) q[8];
cx q[13], q[8];
cx q[23], q[4];
rz(-0.5898352861404419) q[4];
cx q[23], q[4];
cx q[7], q[33];
rz(-0.5898352861404419) q[33];
cx q[7], q[33];
cx q[31], q[21];
rz(-0.5898352861404419) q[21];
cx q[31], q[21];
cx q[12], q[24];
rz(-0.5898352861404419) q[24];
cx q[12], q[24];
cx q[20], q[24];
rz(-0.5898352861404419) q[24];
cx q[20], q[24];
cx q[6], q[31];
rz(-0.5898352861404419) q[31];
cx q[6], q[31];
cx q[0], q[31];
rz(-0.5898352861404419) q[31];
cx q[0], q[31];
cx q[0], q[37];
rz(-0.5898352861404419) q[37];
cx q[0], q[37];
cx q[18], q[19];
rz(-0.5898352861404419) q[19];
cx q[18], q[19];
cx q[28], q[5];
rz(-0.5898352861404419) q[5];
cx q[28], q[5];
cx q[13], q[32];
rz(-0.5898352861404419) q[32];
cx q[13], q[32];
cx q[32], q[3];
rz(-0.5898352861404419) q[3];
cx q[32], q[3];
cx q[31], q[30];
rz(-0.5898352861404419) q[30];
cx q[31], q[30];
cx q[7], q[18];
rz(-0.5898352861404419) q[18];
cx q[7], q[18];
cx q[6], q[3];
rz(-0.5898352861404419) q[3];
cx q[6], q[3];
cx q[34], q[35];
rz(-0.5898352861404419) q[35];
cx q[34], q[35];
cx q[2], q[21];
rz(-0.5898352861404419) q[21];
cx q[2], q[21];
cx q[35], q[7];
rz(-0.5898352861404419) q[7];
cx q[35], q[7];
cx q[1], q[37];
rz(-0.5898352861404419) q[37];
cx q[1], q[37];
cx q[0], q[24];
rz(-0.5898352861404419) q[24];
cx q[0], q[24];
cx q[27], q[29];
rz(-0.5898352861404419) q[29];
cx q[27], q[29];
cx q[7], q[9];
rz(-0.5898352861404419) q[9];
cx q[7], q[9];
cx q[28], q[25];
rz(-0.5898352861404419) q[25];
cx q[28], q[25];
cx q[10], q[24];
rz(-0.5898352861404419) q[24];
cx q[10], q[24];
cx q[13], q[19];
rz(-0.5898352861404419) q[19];
cx q[13], q[19];
cx q[4], q[10];
rz(-0.5898352861404419) q[10];
cx q[4], q[10];
cx q[14], q[12];
rz(-0.5898352861404419) q[12];
cx q[14], q[12];
cx q[24], q[25];
rz(-0.5898352861404419) q[25];
cx q[24], q[25];
cx q[10], q[25];
rz(-0.5898352861404419) q[25];
cx q[10], q[25];
cx q[1], q[12];
rz(-0.5898352861404419) q[12];
cx q[1], q[12];
cx q[21], q[7];
rz(-0.5898352861404419) q[7];
cx q[21], q[7];
cx q[22], q[7];
rz(-0.5898352861404419) q[7];
cx q[22], q[7];
cx q[6], q[37];
rz(-0.5898352861404419) q[37];
cx q[6], q[37];
cx q[10], q[28];
rz(-0.5898352861404419) q[28];
cx q[10], q[28];
cx q[4], q[11];
rz(-0.5898352861404419) q[11];
cx q[4], q[11];
cx q[34], q[15];
rz(-0.5898352861404419) q[15];
cx q[34], q[15];
cx q[37], q[31];
rz(-0.5898352861404419) q[31];
cx q[37], q[31];
cx q[7], q[8];
rz(-0.5898352861404419) q[8];
cx q[7], q[8];
cx q[25], q[22];
rz(-0.5898352861404419) q[22];
cx q[25], q[22];
cx q[22], q[36];
rz(-0.5898352861404419) q[36];
cx q[22], q[36];
cx q[18], q[23];
rz(-0.5898352861404419) q[23];
cx q[18], q[23];
cx q[9], q[0];
rz(-0.5898352861404419) q[0];
cx q[9], q[0];
cx q[28], q[7];
rz(-0.5898352861404419) q[7];
cx q[28], q[7];
cx q[2], q[30];
rz(-0.5898352861404419) q[30];
cx q[2], q[30];
cx q[33], q[4];
rz(-0.5898352861404419) q[4];
cx q[33], q[4];
cx q[8], q[31];
rz(-0.5898352861404419) q[31];
cx q[8], q[31];
cx q[6], q[2];
rz(-0.5898352861404419) q[2];
cx q[6], q[2];
cx q[25], q[3];
rz(-0.5898352861404419) q[3];
cx q[25], q[3];
cx q[30], q[15];
rz(-0.5898352861404419) q[15];
cx q[30], q[15];
cx q[25], q[4];
rz(-0.5898352861404419) q[4];
cx q[25], q[4];
cx q[6], q[14];
rz(-0.5898352861404419) q[14];
cx q[6], q[14];
cx q[4], q[3];
rz(-0.5898352861404419) q[3];
cx q[4], q[3];
cx q[32], q[4];
rz(-0.5898352861404419) q[4];
cx q[32], q[4];
cx q[29], q[12];
rz(-0.5898352861404419) q[12];
cx q[29], q[12];
cx q[13], q[28];
rz(-0.5898352861404419) q[28];
cx q[13], q[28];
cx q[21], q[1];
rz(-0.5898352861404419) q[1];
cx q[21], q[1];
cx q[25], q[13];
rz(-0.5898352861404419) q[13];
cx q[25], q[13];
cx q[35], q[37];
rz(-0.5898352861404419) q[37];
cx q[35], q[37];
cx q[1], q[18];
rz(-0.5898352861404419) q[18];
cx q[1], q[18];
cx q[37], q[28];
rz(-0.5898352861404419) q[28];
cx q[37], q[28];
cx q[14], q[25];
rz(-0.5898352861404419) q[25];
cx q[14], q[25];
cx q[17], q[4];
rz(-0.5898352861404419) q[4];
cx q[17], q[4];
cx q[2], q[22];
rz(-0.5898352861404419) q[22];
cx q[2], q[22];
cx q[18], q[37];
rz(-0.5898352861404419) q[37];
cx q[18], q[37];
cx q[35], q[1];
rz(-0.5898352861404419) q[1];
cx q[35], q[1];
cx q[36], q[25];
rz(-0.5898352861404419) q[25];
cx q[36], q[25];
cx q[24], q[1];
rz(-0.5898352861404419) q[1];
cx q[24], q[1];
cx q[36], q[35];
rz(-0.5898352861404419) q[35];
cx q[36], q[35];
cx q[3], q[17];
rz(-0.5898352861404419) q[17];
cx q[3], q[17];
cx q[21], q[10];
rz(-0.5898352861404419) q[10];
cx q[21], q[10];
cx q[8], q[28];
rz(-0.5898352861404419) q[28];
cx q[8], q[28];
cx q[9], q[3];
rz(-0.5898352861404419) q[3];
cx q[9], q[3];
cx q[0], q[36];
rz(-0.5898352861404419) q[36];
cx q[0], q[36];
cx q[21], q[36];
rz(-0.5898352861404419) q[36];
cx q[21], q[36];
cx q[10], q[33];
rz(-0.5898352861404419) q[33];
cx q[10], q[33];
cx q[13], q[15];
rz(-0.5898352861404419) q[15];
cx q[13], q[15];
cx q[6], q[34];
rz(-0.5898352861404419) q[34];
cx q[6], q[34];
cx q[15], q[19];
rz(-0.5898352861404419) q[19];
cx q[15], q[19];
cx q[10], q[35];
rz(-0.5898352861404419) q[35];
cx q[10], q[35];
cx q[32], q[6];
rz(-0.5898352861404419) q[6];
cx q[32], q[6];
cx q[32], q[14];
rz(-0.5898352861404419) q[14];
cx q[32], q[14];
cx q[15], q[9];
rz(-0.5898352861404419) q[9];
cx q[15], q[9];
cx q[23], q[21];
rz(-0.5898352861404419) q[21];
cx q[23], q[21];
cx q[4], q[20];
rz(-0.5898352861404419) q[20];
cx q[4], q[20];
cx q[24], q[27];
rz(-0.5898352861404419) q[27];
cx q[24], q[27];
cx q[20], q[12];
rz(-0.5898352861404419) q[12];
cx q[20], q[12];
cx q[8], q[4];
rz(-0.5898352861404419) q[4];
cx q[8], q[4];
cx q[34], q[14];
rz(-0.5898352861404419) q[14];
cx q[34], q[14];
cx q[0], q[19];
rz(-0.5898352861404419) q[19];
cx q[0], q[19];
cx q[7], q[26];
rz(-0.5898352861404419) q[26];
cx q[7], q[26];
cx q[13], q[29];
rz(-0.5898352861404419) q[29];
cx q[13], q[29];
cx q[29], q[28];
rz(-0.5898352861404419) q[28];
cx q[29], q[28];
cx q[20], q[29];
rz(-0.5898352861404419) q[29];
cx q[20], q[29];
cx q[32], q[31];
rz(-0.5898352861404419) q[31];
cx q[32], q[31];
cx q[4], q[16];
rz(-0.5898352861404419) q[16];
cx q[4], q[16];
cx q[15], q[27];
rz(-0.5898352861404419) q[27];
cx q[15], q[27];
cx q[5], q[21];
rz(-0.5898352861404419) q[21];
cx q[5], q[21];
cx q[30], q[27];
rz(-0.5898352861404419) q[27];
cx q[30], q[27];
cx q[5], q[12];
rz(-0.5898352861404419) q[12];
cx q[5], q[12];
cx q[33], q[8];
rz(-0.5898352861404419) q[8];
cx q[33], q[8];
cx q[13], q[17];
rz(-0.5898352861404419) q[17];
cx q[13], q[17];
cx q[25], q[8];
rz(-0.5898352861404419) q[8];
cx q[25], q[8];
cx q[10], q[29];
rz(-0.5898352861404419) q[29];
cx q[10], q[29];
cx q[27], q[35];
rz(-0.5898352861404419) q[35];
cx q[27], q[35];
cx q[1], q[13];
rz(-0.5898352861404419) q[13];
cx q[1], q[13];
cx q[29], q[19];
rz(-0.5898352861404419) q[19];
cx q[29], q[19];
cx q[20], q[23];
rz(-0.5898352861404419) q[23];
cx q[20], q[23];
cx q[17], q[6];
rz(-0.5898352861404419) q[6];
cx q[17], q[6];
cx q[18], q[9];
rz(-0.5898352861404419) q[9];
cx q[18], q[9];
cx q[36], q[13];
rz(-0.5898352861404419) q[13];
cx q[36], q[13];
cx q[35], q[18];
rz(-0.5898352861404419) q[18];
cx q[35], q[18];
cx q[36], q[32];
rz(-0.5898352861404419) q[32];
cx q[36], q[32];
cx q[24], q[19];
rz(-0.5898352861404419) q[19];
cx q[24], q[19];
cx q[29], q[35];
rz(-0.5898352861404419) q[35];
cx q[29], q[35];
cx q[27], q[34];
rz(-0.5898352861404419) q[34];
cx q[27], q[34];
cx q[23], q[35];
rz(-0.5898352861404419) q[35];
cx q[23], q[35];
cx q[3], q[30];
rz(-0.5898352861404419) q[30];
cx q[3], q[30];
cx q[20], q[33];
rz(-0.5898352861404419) q[33];
cx q[20], q[33];
cx q[15], q[5];
rz(-0.5898352861404419) q[5];
cx q[15], q[5];
cx q[37], q[23];
rz(-0.5898352861404419) q[23];
cx q[37], q[23];
cx q[0], q[13];
rz(-0.5898352861404419) q[13];
cx q[0], q[13];
cx q[21], q[35];
rz(-0.5898352861404419) q[35];
cx q[21], q[35];
cx q[9], q[13];
rz(-0.5898352861404419) q[13];
cx q[9], q[13];
cx q[17], q[33];
rz(-0.5898352861404419) q[33];
cx q[17], q[33];
cx q[33], q[16];
rz(-0.5898352861404419) q[16];
cx q[33], q[16];
cx q[22], q[26];
rz(-0.5898352861404419) q[26];
cx q[22], q[26];
cx q[30], q[18];
rz(-0.5898352861404419) q[18];
cx q[30], q[18];
cx q[7], q[17];
rz(-0.5898352861404419) q[17];
cx q[7], q[17];
cx q[11], q[5];
rz(-0.5898352861404419) q[5];
cx q[11], q[5];
cx q[7], q[10];
rz(-0.5898352861404419) q[10];
cx q[7], q[10];
cx q[21], q[28];
rz(-0.5898352861404419) q[28];
cx q[21], q[28];
cx q[10], q[37];
rz(-0.5898352861404419) q[37];
cx q[10], q[37];
cx q[24], q[5];
rz(-0.5898352861404419) q[5];
cx q[24], q[5];
cx q[15], q[6];
rz(-0.5898352861404419) q[6];
cx q[15], q[6];
cx q[7], q[11];
rz(-0.5898352861404419) q[11];
cx q[7], q[11];
cx q[34], q[8];
rz(-0.5898352861404419) q[8];
cx q[34], q[8];
cx q[35], q[11];
rz(-0.5898352861404419) q[11];
cx q[35], q[11];
cx q[30], q[10];
rz(-0.5898352861404419) q[10];
cx q[30], q[10];
cx q[10], q[19];
rz(-0.5898352861404419) q[19];
cx q[10], q[19];
cx q[24], q[32];
rz(-0.5898352861404419) q[32];
cx q[24], q[32];
cx q[29], q[30];
rz(-0.5898352861404419) q[30];
cx q[29], q[30];
cx q[26], q[33];
rz(-0.5898352861404419) q[33];
cx q[26], q[33];
cx q[3], q[24];
rz(-0.5898352861404419) q[24];
cx q[3], q[24];
cx q[25], q[17];
rz(-0.5898352861404419) q[17];
cx q[25], q[17];
cx q[8], q[22];
rz(-0.5898352861404419) q[22];
cx q[8], q[22];
cx q[18], q[6];
rz(-0.5898352861404419) q[6];
cx q[18], q[6];
cx q[12], q[16];
rz(-0.5898352861404419) q[16];
cx q[12], q[16];
cx q[16], q[5];
rz(-0.5898352861404419) q[5];
cx q[16], q[5];
cx q[26], q[8];
rz(-0.5898352861404419) q[8];
cx q[26], q[8];
cx q[28], q[31];
rz(-0.5898352861404419) q[31];
cx q[28], q[31];
cx q[15], q[28];
rz(-0.5898352861404419) q[28];
cx q[15], q[28];
cx q[30], q[20];
rz(-0.5898352861404419) q[20];
cx q[30], q[20];
cx q[31], q[27];
rz(-0.5898352861404419) q[27];
cx q[31], q[27];
cx q[0], q[15];
rz(-0.5898352861404419) q[15];
cx q[0], q[15];
cx q[16], q[29];
rz(-0.5898352861404419) q[29];
cx q[16], q[29];
cx q[24], q[34];
rz(-0.5898352861404419) q[34];
cx q[24], q[34];
cx q[4], q[14];
rz(-0.5898352861404419) q[14];
cx q[4], q[14];
cx q[5], q[36];
rz(-0.5898352861404419) q[36];
cx q[5], q[36];
cx q[30], q[36];
rz(-0.5898352861404419) q[36];
cx q[30], q[36];
cx q[26], q[31];
rz(-0.5898352861404419) q[31];
cx q[26], q[31];
cx q[36], q[37];
rz(-0.5898352861404419) q[37];
cx q[36], q[37];
cx q[9], q[26];
rz(-0.5898352861404419) q[26];
cx q[9], q[26];
cx q[2], q[9];
rz(-0.5898352861404419) q[9];
cx q[2], q[9];
cx q[31], q[22];
rz(-0.5898352861404419) q[22];
cx q[31], q[22];
cx q[4], q[29];
rz(-0.5898352861404419) q[29];
cx q[4], q[29];
cx q[19], q[1];
rz(-0.5898352861404419) q[1];
cx q[19], q[1];
cx q[29], q[17];
rz(-0.5898352861404419) q[17];
cx q[29], q[17];
cx q[0], q[26];
rz(-0.5898352861404419) q[26];
cx q[0], q[26];
cx q[18], q[20];
rz(-0.5898352861404419) q[20];
cx q[18], q[20];
cx q[15], q[35];
rz(-0.5898352861404419) q[35];
cx q[15], q[35];
cx q[13], q[21];
rz(-0.5898352861404419) q[21];
cx q[13], q[21];
cx q[20], q[34];
rz(-0.5898352861404419) q[34];
cx q[20], q[34];
cx q[24], q[22];
rz(-0.5898352861404419) q[22];
cx q[24], q[22];
cx q[22], q[17];
rz(-0.5898352861404419) q[17];
cx q[22], q[17];
cx q[1], q[4];
rz(-0.5898352861404419) q[4];
cx q[1], q[4];
cx q[24], q[33];
rz(-0.5898352861404419) q[33];
cx q[24], q[33];
cx q[11], q[8];
rz(-0.5898352861404419) q[8];
cx q[11], q[8];
cx q[34], q[33];
rz(-0.5898352861404419) q[33];
cx q[34], q[33];
cx q[36], q[17];
rz(-0.5898352861404419) q[17];
cx q[36], q[17];
cx q[9], q[6];
rz(-0.5898352861404419) q[6];
cx q[9], q[6];
cx q[27], q[33];
rz(-0.5898352861404419) q[33];
cx q[27], q[33];
cx q[29], q[18];
rz(-0.5898352861404419) q[18];
cx q[29], q[18];
cx q[12], q[9];
rz(-0.5898352861404419) q[9];
cx q[12], q[9];
cx q[2], q[11];
rz(-0.5898352861404419) q[11];
cx q[2], q[11];
cx q[36], q[19];
rz(-0.5898352861404419) q[19];
cx q[36], q[19];
cx q[31], q[29];
rz(-0.5898352861404419) q[29];
cx q[31], q[29];
cx q[23], q[30];
rz(-0.5898352861404419) q[30];
cx q[23], q[30];
cx q[30], q[24];
rz(-0.5898352861404419) q[24];
cx q[30], q[24];
cx q[27], q[20];
rz(-0.5898352861404419) q[20];
cx q[27], q[20];
cx q[22], q[34];
rz(-0.5898352861404419) q[34];
cx q[22], q[34];
cx q[9], q[1];
rz(-0.5898352861404419) q[1];
cx q[9], q[1];
cx q[22], q[4];
rz(-0.5898352861404419) q[4];
cx q[22], q[4];
cx q[28], q[17];
rz(-0.5898352861404419) q[17];
cx q[28], q[17];
cx q[13], q[33];
rz(-0.5898352861404419) q[33];
cx q[13], q[33];
cx q[24], q[23];
rz(-0.5898352861404419) q[23];
cx q[24], q[23];
cx q[23], q[16];
rz(-0.5898352861404419) q[16];
cx q[23], q[16];
cx q[14], q[19];
rz(-0.5898352861404419) q[19];
cx q[14], q[19];
cx q[0], q[32];
rz(-0.5898352861404419) q[32];
cx q[0], q[32];
cx q[18], q[10];
rz(-0.5898352861404419) q[10];
cx q[18], q[10];
rx(1.690858244895935) q[0];
rx(1.690858244895935) q[1];
rx(1.690858244895935) q[2];
rx(1.690858244895935) q[3];
rx(1.690858244895935) q[4];
rx(1.690858244895935) q[5];
rx(1.690858244895935) q[6];
rx(1.690858244895935) q[7];
rx(1.690858244895935) q[8];
rx(1.690858244895935) q[9];
rx(1.690858244895935) q[10];
rx(1.690858244895935) q[11];
rx(1.690858244895935) q[12];
rx(1.690858244895935) q[13];
rx(1.690858244895935) q[14];
rx(1.690858244895935) q[15];
rx(1.690858244895935) q[16];
rx(1.690858244895935) q[17];
rx(1.690858244895935) q[18];
rx(1.690858244895935) q[19];
rx(1.690858244895935) q[20];
rx(1.690858244895935) q[21];
rx(1.690858244895935) q[22];
rx(1.690858244895935) q[23];
rx(1.690858244895935) q[24];
rx(1.690858244895935) q[25];
rx(1.690858244895935) q[26];
rx(1.690858244895935) q[27];
rx(1.690858244895935) q[28];
rx(1.690858244895935) q[29];
rx(1.690858244895935) q[30];
rx(1.690858244895935) q[31];
rx(1.690858244895935) q[32];
rx(1.690858244895935) q[33];
rx(1.690858244895935) q[34];
rx(1.690858244895935) q[35];
rx(1.690858244895935) q[36];
rx(1.690858244895935) q[37];