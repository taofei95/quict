OPENQASM 2.0;
include "qelib1.inc";
qreg q[79];
creg c[77];
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
x q[77];
h q[77];
x q[0];
x q[1];
x q[4];
x q[5];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[14];
x q[16];
x q[18];
x q[19];
x q[20];
x q[25];
x q[26];
x q[29];
x q[30];
x q[31];
x q[32];
x q[34];
x q[37];
x q[39];
x q[40];
x q[41];
x q[42];
x q[43];
x q[46];
x q[47];
x q[49];
x q[50];
x q[52];
x q[53];
x q[55];
x q[57];
x q[60];
x q[62];
x q[63];
x q[67];
x q[68];
x q[70];
x q[73];
x q[74];
ccx q[38], q[77], q[78];
ccx q[37], q[76], q[77];
ccx q[36], q[75], q[76];
ccx q[35], q[74], q[75];
ccx q[34], q[73], q[74];
ccx q[33], q[72], q[73];
ccx q[32], q[71], q[72];
ccx q[31], q[70], q[71];
ccx q[30], q[69], q[70];
ccx q[29], q[68], q[69];
ccx q[28], q[67], q[68];
ccx q[27], q[66], q[67];
ccx q[26], q[65], q[66];
ccx q[25], q[64], q[65];
ccx q[24], q[63], q[64];
ccx q[23], q[62], q[63];
ccx q[22], q[61], q[62];
ccx q[21], q[60], q[61];
ccx q[20], q[59], q[60];
ccx q[19], q[58], q[59];
ccx q[18], q[57], q[58];
ccx q[17], q[56], q[57];
ccx q[16], q[55], q[56];
ccx q[15], q[54], q[55];
ccx q[14], q[53], q[54];
ccx q[13], q[52], q[53];
ccx q[12], q[51], q[52];
ccx q[11], q[50], q[51];
ccx q[10], q[49], q[50];
ccx q[9], q[48], q[49];
ccx q[8], q[47], q[48];
ccx q[7], q[46], q[47];
ccx q[6], q[45], q[46];
ccx q[5], q[44], q[45];
ccx q[4], q[43], q[44];
ccx q[3], q[42], q[43];
ccx q[2], q[41], q[42];
ccx q[0], q[1], q[41];
ccx q[2], q[41], q[42];
ccx q[3], q[42], q[43];
ccx q[4], q[43], q[44];
ccx q[5], q[44], q[45];
ccx q[6], q[45], q[46];
ccx q[7], q[46], q[47];
ccx q[8], q[47], q[48];
ccx q[9], q[48], q[49];
ccx q[10], q[49], q[50];
ccx q[11], q[50], q[51];
ccx q[12], q[51], q[52];
ccx q[13], q[52], q[53];
ccx q[14], q[53], q[54];
ccx q[15], q[54], q[55];
ccx q[16], q[55], q[56];
ccx q[17], q[56], q[57];
ccx q[18], q[57], q[58];
ccx q[19], q[58], q[59];
ccx q[20], q[59], q[60];
ccx q[21], q[60], q[61];
ccx q[22], q[61], q[62];
ccx q[23], q[62], q[63];
ccx q[24], q[63], q[64];
ccx q[25], q[64], q[65];
ccx q[26], q[65], q[66];
ccx q[27], q[66], q[67];
ccx q[28], q[67], q[68];
ccx q[29], q[68], q[69];
ccx q[30], q[69], q[70];
ccx q[31], q[70], q[71];
ccx q[32], q[71], q[72];
ccx q[33], q[72], q[73];
ccx q[34], q[73], q[74];
ccx q[35], q[74], q[75];
ccx q[36], q[75], q[76];
ccx q[37], q[76], q[77];
ccx q[38], q[77], q[78];
ccx q[37], q[76], q[77];
ccx q[36], q[75], q[76];
ccx q[35], q[74], q[75];
ccx q[34], q[73], q[74];
ccx q[33], q[72], q[73];
ccx q[32], q[71], q[72];
ccx q[31], q[70], q[71];
ccx q[30], q[69], q[70];
ccx q[29], q[68], q[69];
ccx q[28], q[67], q[68];
ccx q[27], q[66], q[67];
ccx q[26], q[65], q[66];
ccx q[25], q[64], q[65];
ccx q[24], q[63], q[64];
ccx q[23], q[62], q[63];
ccx q[22], q[61], q[62];
ccx q[21], q[60], q[61];
ccx q[20], q[59], q[60];
ccx q[19], q[58], q[59];
ccx q[18], q[57], q[58];
ccx q[17], q[56], q[57];
ccx q[16], q[55], q[56];
ccx q[15], q[54], q[55];
ccx q[14], q[53], q[54];
ccx q[13], q[52], q[53];
ccx q[12], q[51], q[52];
ccx q[11], q[50], q[51];
ccx q[10], q[49], q[50];
ccx q[9], q[48], q[49];
ccx q[8], q[47], q[48];
ccx q[7], q[46], q[47];
ccx q[6], q[45], q[46];
ccx q[5], q[44], q[45];
ccx q[4], q[43], q[44];
ccx q[3], q[42], q[43];
ccx q[2], q[41], q[42];
ccx q[0], q[1], q[41];
ccx q[2], q[41], q[42];
ccx q[3], q[42], q[43];
ccx q[4], q[43], q[44];
ccx q[5], q[44], q[45];
ccx q[6], q[45], q[46];
ccx q[7], q[46], q[47];
ccx q[8], q[47], q[48];
ccx q[9], q[48], q[49];
ccx q[10], q[49], q[50];
ccx q[11], q[50], q[51];
ccx q[12], q[51], q[52];
ccx q[13], q[52], q[53];
ccx q[14], q[53], q[54];
ccx q[15], q[54], q[55];
ccx q[16], q[55], q[56];
ccx q[17], q[56], q[57];
ccx q[18], q[57], q[58];
ccx q[19], q[58], q[59];
ccx q[20], q[59], q[60];
ccx q[21], q[60], q[61];
ccx q[22], q[61], q[62];
ccx q[23], q[62], q[63];
ccx q[24], q[63], q[64];
ccx q[25], q[64], q[65];
ccx q[26], q[65], q[66];
ccx q[27], q[66], q[67];
ccx q[28], q[67], q[68];
ccx q[29], q[68], q[69];
ccx q[30], q[69], q[70];
ccx q[31], q[70], q[71];
ccx q[32], q[71], q[72];
ccx q[33], q[72], q[73];
ccx q[34], q[73], q[74];
ccx q[35], q[74], q[75];
ccx q[36], q[75], q[76];
ccx q[37], q[76], q[77];
h q[77];
s q[78];
ccx q[77], q[38], q[78];
ccx q[76], q[37], q[38];
ccx q[75], q[36], q[37];
ccx q[74], q[35], q[36];
ccx q[73], q[34], q[35];
ccx q[72], q[33], q[34];
ccx q[71], q[32], q[33];
ccx q[70], q[31], q[32];
ccx q[69], q[30], q[31];
ccx q[68], q[29], q[30];
ccx q[67], q[28], q[29];
ccx q[66], q[27], q[28];
ccx q[65], q[26], q[27];
ccx q[64], q[25], q[26];
ccx q[63], q[24], q[25];
ccx q[62], q[23], q[24];
ccx q[61], q[22], q[23];
ccx q[60], q[21], q[22];
ccx q[59], q[20], q[21];
ccx q[58], q[19], q[20];
ccx q[57], q[18], q[19];
ccx q[56], q[17], q[18];
ccx q[55], q[16], q[17];
ccx q[54], q[15], q[16];
ccx q[53], q[14], q[15];
ccx q[52], q[13], q[14];
ccx q[51], q[12], q[13];
ccx q[50], q[11], q[12];
ccx q[49], q[10], q[11];
ccx q[48], q[9], q[10];
ccx q[47], q[8], q[9];
ccx q[46], q[7], q[8];
ccx q[45], q[6], q[7];
ccx q[44], q[5], q[6];
ccx q[43], q[4], q[5];
ccx q[42], q[3], q[4];
ccx q[41], q[2], q[3];
ccx q[39], q[40], q[2];
ccx q[41], q[2], q[3];
ccx q[42], q[3], q[4];
ccx q[43], q[4], q[5];
ccx q[44], q[5], q[6];
ccx q[45], q[6], q[7];
ccx q[46], q[7], q[8];
ccx q[47], q[8], q[9];
ccx q[48], q[9], q[10];
ccx q[49], q[10], q[11];
ccx q[50], q[11], q[12];
ccx q[51], q[12], q[13];
ccx q[52], q[13], q[14];
ccx q[53], q[14], q[15];
ccx q[54], q[15], q[16];
ccx q[55], q[16], q[17];
ccx q[56], q[17], q[18];
ccx q[57], q[18], q[19];
ccx q[58], q[19], q[20];
ccx q[59], q[20], q[21];
ccx q[60], q[21], q[22];
ccx q[61], q[22], q[23];
ccx q[62], q[23], q[24];
ccx q[63], q[24], q[25];
ccx q[64], q[25], q[26];
ccx q[65], q[26], q[27];
ccx q[66], q[27], q[28];
ccx q[67], q[28], q[29];
ccx q[68], q[29], q[30];
ccx q[69], q[30], q[31];
ccx q[70], q[31], q[32];
ccx q[71], q[32], q[33];
ccx q[72], q[33], q[34];
ccx q[73], q[34], q[35];
ccx q[74], q[35], q[36];
ccx q[75], q[36], q[37];
ccx q[76], q[37], q[38];
ccx q[77], q[38], q[78];
ccx q[76], q[37], q[38];
ccx q[75], q[36], q[37];
ccx q[74], q[35], q[36];
ccx q[73], q[34], q[35];
ccx q[72], q[33], q[34];
ccx q[71], q[32], q[33];
ccx q[70], q[31], q[32];
ccx q[69], q[30], q[31];
ccx q[68], q[29], q[30];
ccx q[67], q[28], q[29];
ccx q[66], q[27], q[28];
ccx q[65], q[26], q[27];
ccx q[64], q[25], q[26];
ccx q[63], q[24], q[25];
ccx q[62], q[23], q[24];
ccx q[61], q[22], q[23];
ccx q[60], q[21], q[22];
ccx q[59], q[20], q[21];
ccx q[58], q[19], q[20];
ccx q[57], q[18], q[19];
ccx q[56], q[17], q[18];
ccx q[55], q[16], q[17];
ccx q[54], q[15], q[16];
ccx q[53], q[14], q[15];
ccx q[52], q[13], q[14];
ccx q[51], q[12], q[13];
ccx q[50], q[11], q[12];
ccx q[49], q[10], q[11];
ccx q[48], q[9], q[10];
ccx q[47], q[8], q[9];
ccx q[46], q[7], q[8];
ccx q[45], q[6], q[7];
ccx q[44], q[5], q[6];
ccx q[43], q[4], q[5];
ccx q[42], q[3], q[4];
ccx q[41], q[2], q[3];
ccx q[39], q[40], q[2];
ccx q[41], q[2], q[3];
ccx q[42], q[3], q[4];
ccx q[43], q[4], q[5];
ccx q[44], q[5], q[6];
ccx q[45], q[6], q[7];
ccx q[46], q[7], q[8];
ccx q[47], q[8], q[9];
ccx q[48], q[9], q[10];
ccx q[49], q[10], q[11];
ccx q[50], q[11], q[12];
ccx q[51], q[12], q[13];
ccx q[52], q[13], q[14];
ccx q[53], q[14], q[15];
ccx q[54], q[15], q[16];
ccx q[55], q[16], q[17];
ccx q[56], q[17], q[18];
ccx q[57], q[18], q[19];
ccx q[58], q[19], q[20];
ccx q[59], q[20], q[21];
ccx q[60], q[21], q[22];
ccx q[61], q[22], q[23];
ccx q[62], q[23], q[24];
ccx q[63], q[24], q[25];
ccx q[64], q[25], q[26];
ccx q[65], q[26], q[27];
ccx q[66], q[27], q[28];
ccx q[67], q[28], q[29];
ccx q[68], q[29], q[30];
ccx q[69], q[30], q[31];
ccx q[70], q[31], q[32];
ccx q[71], q[32], q[33];
ccx q[72], q[33], q[34];
ccx q[73], q[34], q[35];
ccx q[74], q[35], q[36];
ccx q[75], q[36], q[37];
ccx q[76], q[37], q[38];
sdg q[78];
ccx q[38], q[77], q[78];
ccx q[37], q[76], q[77];
ccx q[36], q[75], q[76];
ccx q[35], q[74], q[75];
ccx q[34], q[73], q[74];
ccx q[33], q[72], q[73];
ccx q[32], q[71], q[72];
ccx q[31], q[70], q[71];
ccx q[30], q[69], q[70];
ccx q[29], q[68], q[69];
ccx q[28], q[67], q[68];
ccx q[27], q[66], q[67];
ccx q[26], q[65], q[66];
ccx q[25], q[64], q[65];
ccx q[24], q[63], q[64];
ccx q[23], q[62], q[63];
ccx q[22], q[61], q[62];
ccx q[21], q[60], q[61];
ccx q[20], q[59], q[60];
ccx q[19], q[58], q[59];
ccx q[18], q[57], q[58];
ccx q[17], q[56], q[57];
ccx q[16], q[55], q[56];
ccx q[15], q[54], q[55];
ccx q[14], q[53], q[54];
ccx q[13], q[52], q[53];
ccx q[12], q[51], q[52];
ccx q[11], q[50], q[51];
ccx q[10], q[49], q[50];
ccx q[9], q[48], q[49];
ccx q[8], q[47], q[48];
ccx q[7], q[46], q[47];
ccx q[6], q[45], q[46];
ccx q[5], q[44], q[45];
ccx q[4], q[43], q[44];
ccx q[3], q[42], q[43];
ccx q[2], q[41], q[42];
ccx q[0], q[1], q[41];
ccx q[2], q[41], q[42];
ccx q[3], q[42], q[43];
ccx q[4], q[43], q[44];
ccx q[5], q[44], q[45];
ccx q[6], q[45], q[46];
ccx q[7], q[46], q[47];
ccx q[8], q[47], q[48];
ccx q[9], q[48], q[49];
ccx q[10], q[49], q[50];
ccx q[11], q[50], q[51];
ccx q[12], q[51], q[52];
ccx q[13], q[52], q[53];
ccx q[14], q[53], q[54];
ccx q[15], q[54], q[55];
ccx q[16], q[55], q[56];
ccx q[17], q[56], q[57];
ccx q[18], q[57], q[58];
ccx q[19], q[58], q[59];
ccx q[20], q[59], q[60];
ccx q[21], q[60], q[61];
ccx q[22], q[61], q[62];
ccx q[23], q[62], q[63];
ccx q[24], q[63], q[64];
ccx q[25], q[64], q[65];
ccx q[26], q[65], q[66];
ccx q[27], q[66], q[67];
ccx q[28], q[67], q[68];
ccx q[29], q[68], q[69];
ccx q[30], q[69], q[70];
ccx q[31], q[70], q[71];
ccx q[32], q[71], q[72];
ccx q[33], q[72], q[73];
ccx q[34], q[73], q[74];
ccx q[35], q[74], q[75];
ccx q[36], q[75], q[76];
ccx q[37], q[76], q[77];
ccx q[38], q[77], q[78];
ccx q[37], q[76], q[77];
ccx q[36], q[75], q[76];
ccx q[35], q[74], q[75];
ccx q[34], q[73], q[74];
ccx q[33], q[72], q[73];
ccx q[32], q[71], q[72];
ccx q[31], q[70], q[71];
ccx q[30], q[69], q[70];
ccx q[29], q[68], q[69];
ccx q[28], q[67], q[68];
ccx q[27], q[66], q[67];
ccx q[26], q[65], q[66];
ccx q[25], q[64], q[65];
ccx q[24], q[63], q[64];
ccx q[23], q[62], q[63];
ccx q[22], q[61], q[62];
ccx q[21], q[60], q[61];
ccx q[20], q[59], q[60];
ccx q[19], q[58], q[59];
ccx q[18], q[57], q[58];
ccx q[17], q[56], q[57];
ccx q[16], q[55], q[56];
ccx q[15], q[54], q[55];
ccx q[14], q[53], q[54];
ccx q[13], q[52], q[53];
ccx q[12], q[51], q[52];
ccx q[11], q[50], q[51];
ccx q[10], q[49], q[50];
ccx q[9], q[48], q[49];
ccx q[8], q[47], q[48];
ccx q[7], q[46], q[47];
ccx q[6], q[45], q[46];
ccx q[5], q[44], q[45];
ccx q[4], q[43], q[44];
ccx q[3], q[42], q[43];
ccx q[2], q[41], q[42];
ccx q[0], q[1], q[41];
ccx q[2], q[41], q[42];
ccx q[3], q[42], q[43];
ccx q[4], q[43], q[44];
ccx q[5], q[44], q[45];
ccx q[6], q[45], q[46];
ccx q[7], q[46], q[47];
ccx q[8], q[47], q[48];
ccx q[9], q[48], q[49];
ccx q[10], q[49], q[50];
ccx q[11], q[50], q[51];
ccx q[12], q[51], q[52];
ccx q[13], q[52], q[53];
ccx q[14], q[53], q[54];
ccx q[15], q[54], q[55];
ccx q[16], q[55], q[56];
ccx q[17], q[56], q[57];
ccx q[18], q[57], q[58];
ccx q[19], q[58], q[59];
ccx q[20], q[59], q[60];
ccx q[21], q[60], q[61];
ccx q[22], q[61], q[62];
ccx q[23], q[62], q[63];
ccx q[24], q[63], q[64];
ccx q[25], q[64], q[65];
ccx q[26], q[65], q[66];
ccx q[27], q[66], q[67];
ccx q[28], q[67], q[68];
ccx q[29], q[68], q[69];
ccx q[30], q[69], q[70];
ccx q[31], q[70], q[71];
ccx q[32], q[71], q[72];
ccx q[33], q[72], q[73];
ccx q[34], q[73], q[74];
ccx q[35], q[74], q[75];
ccx q[36], q[75], q[76];
ccx q[37], q[76], q[77];
s q[78];
ccx q[77], q[38], q[78];
ccx q[76], q[37], q[38];
ccx q[75], q[36], q[37];
ccx q[74], q[35], q[36];
ccx q[73], q[34], q[35];
ccx q[72], q[33], q[34];
ccx q[71], q[32], q[33];
ccx q[70], q[31], q[32];
ccx q[69], q[30], q[31];
ccx q[68], q[29], q[30];
ccx q[67], q[28], q[29];
ccx q[66], q[27], q[28];
ccx q[65], q[26], q[27];
ccx q[64], q[25], q[26];
ccx q[63], q[24], q[25];
ccx q[62], q[23], q[24];
ccx q[61], q[22], q[23];
ccx q[60], q[21], q[22];
ccx q[59], q[20], q[21];
ccx q[58], q[19], q[20];
ccx q[57], q[18], q[19];
ccx q[56], q[17], q[18];
ccx q[55], q[16], q[17];
ccx q[54], q[15], q[16];
ccx q[53], q[14], q[15];
ccx q[52], q[13], q[14];
ccx q[51], q[12], q[13];
ccx q[50], q[11], q[12];
ccx q[49], q[10], q[11];
ccx q[48], q[9], q[10];
ccx q[47], q[8], q[9];
ccx q[46], q[7], q[8];
ccx q[45], q[6], q[7];
ccx q[44], q[5], q[6];
ccx q[43], q[4], q[5];
ccx q[42], q[3], q[4];
ccx q[41], q[2], q[3];
ccx q[39], q[40], q[2];
ccx q[41], q[2], q[3];
ccx q[42], q[3], q[4];
ccx q[43], q[4], q[5];
ccx q[44], q[5], q[6];
ccx q[45], q[6], q[7];
ccx q[46], q[7], q[8];
ccx q[47], q[8], q[9];
ccx q[48], q[9], q[10];
ccx q[49], q[10], q[11];
ccx q[50], q[11], q[12];
ccx q[51], q[12], q[13];
ccx q[52], q[13], q[14];
ccx q[53], q[14], q[15];
ccx q[54], q[15], q[16];
ccx q[55], q[16], q[17];
ccx q[56], q[17], q[18];
ccx q[57], q[18], q[19];
ccx q[58], q[19], q[20];
ccx q[59], q[20], q[21];
ccx q[60], q[21], q[22];
ccx q[61], q[22], q[23];
ccx q[62], q[23], q[24];
ccx q[63], q[24], q[25];
ccx q[64], q[25], q[26];
ccx q[65], q[26], q[27];
ccx q[66], q[27], q[28];
ccx q[67], q[28], q[29];
ccx q[68], q[29], q[30];
ccx q[69], q[30], q[31];
ccx q[70], q[31], q[32];
ccx q[71], q[32], q[33];
ccx q[72], q[33], q[34];
ccx q[73], q[34], q[35];
ccx q[74], q[35], q[36];
ccx q[75], q[36], q[37];
ccx q[76], q[37], q[38];
ccx q[77], q[38], q[78];
ccx q[76], q[37], q[38];
ccx q[75], q[36], q[37];
ccx q[74], q[35], q[36];
ccx q[73], q[34], q[35];
ccx q[72], q[33], q[34];
ccx q[71], q[32], q[33];
ccx q[70], q[31], q[32];
ccx q[69], q[30], q[31];
ccx q[68], q[29], q[30];
ccx q[67], q[28], q[29];
ccx q[66], q[27], q[28];
ccx q[65], q[26], q[27];
ccx q[64], q[25], q[26];
ccx q[63], q[24], q[25];
ccx q[62], q[23], q[24];
ccx q[61], q[22], q[23];
ccx q[60], q[21], q[22];
ccx q[59], q[20], q[21];
ccx q[58], q[19], q[20];
ccx q[57], q[18], q[19];
ccx q[56], q[17], q[18];
ccx q[55], q[16], q[17];
ccx q[54], q[15], q[16];
ccx q[53], q[14], q[15];
ccx q[52], q[13], q[14];
ccx q[51], q[12], q[13];
ccx q[50], q[11], q[12];
ccx q[49], q[10], q[11];
ccx q[48], q[9], q[10];
ccx q[47], q[8], q[9];
ccx q[46], q[7], q[8];
ccx q[45], q[6], q[7];
ccx q[44], q[5], q[6];
ccx q[43], q[4], q[5];
ccx q[42], q[3], q[4];
ccx q[41], q[2], q[3];
ccx q[39], q[40], q[2];
ccx q[41], q[2], q[3];
ccx q[42], q[3], q[4];
ccx q[43], q[4], q[5];
ccx q[44], q[5], q[6];
ccx q[45], q[6], q[7];
ccx q[46], q[7], q[8];
ccx q[47], q[8], q[9];
ccx q[48], q[9], q[10];
ccx q[49], q[10], q[11];
ccx q[50], q[11], q[12];
ccx q[51], q[12], q[13];
ccx q[52], q[13], q[14];
ccx q[53], q[14], q[15];
ccx q[54], q[15], q[16];
ccx q[55], q[16], q[17];
ccx q[56], q[17], q[18];
ccx q[57], q[18], q[19];
ccx q[58], q[19], q[20];
ccx q[59], q[20], q[21];
ccx q[60], q[21], q[22];
ccx q[61], q[22], q[23];
ccx q[62], q[23], q[24];
ccx q[63], q[24], q[25];
ccx q[64], q[25], q[26];
ccx q[65], q[26], q[27];
ccx q[66], q[27], q[28];
ccx q[67], q[28], q[29];
ccx q[68], q[29], q[30];
ccx q[69], q[30], q[31];
ccx q[70], q[31], q[32];
ccx q[71], q[32], q[33];
ccx q[72], q[33], q[34];
ccx q[73], q[34], q[35];
ccx q[74], q[35], q[36];
ccx q[75], q[36], q[37];
ccx q[76], q[37], q[38];
h q[77];
sdg q[78];
x q[0];
x q[1];
x q[4];
x q[5];
x q[6];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[14];
x q[16];
x q[18];
x q[19];
x q[20];
x q[25];
x q[26];
x q[29];
x q[30];
x q[31];
x q[32];
x q[34];
x q[37];
x q[39];
x q[40];
x q[41];
x q[42];
x q[43];
x q[46];
x q[47];
x q[49];
x q[50];
x q[52];
x q[53];
x q[55];
x q[57];
x q[60];
x q[62];
x q[63];
x q[67];
x q[68];
x q[70];
x q[73];
x q[74];
h q[77];
x q[77];
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
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
x q[6];
x q[7];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[13];
x q[14];
x q[15];
x q[16];
x q[17];
x q[18];
x q[19];
x q[20];
x q[21];
x q[22];
x q[23];
x q[24];
x q[25];
x q[26];
x q[27];
x q[28];
x q[29];
x q[30];
x q[31];
x q[32];
x q[33];
x q[34];
x q[35];
x q[36];
x q[37];
x q[38];
x q[39];
x q[40];
x q[41];
x q[42];
x q[43];
x q[44];
x q[45];
x q[46];
x q[47];
x q[48];
x q[49];
x q[50];
x q[51];
x q[52];
x q[53];
x q[54];
x q[55];
x q[56];
x q[57];
x q[58];
x q[59];
x q[60];
x q[61];
x q[62];
x q[63];
x q[64];
x q[65];
x q[66];
x q[67];
x q[68];
x q[69];
x q[70];
x q[71];
x q[72];
x q[73];
x q[74];
x q[75];
x q[76];
h q[76];
ccx q[38], q[76], q[77];
ccx q[37], q[75], q[76];
ccx q[36], q[74], q[75];
ccx q[35], q[73], q[74];
ccx q[34], q[72], q[73];
ccx q[33], q[71], q[72];
ccx q[32], q[70], q[71];
ccx q[31], q[69], q[70];
ccx q[30], q[68], q[69];
ccx q[29], q[67], q[68];
ccx q[28], q[66], q[67];
ccx q[27], q[65], q[66];
ccx q[26], q[64], q[65];
ccx q[25], q[63], q[64];
ccx q[24], q[62], q[63];
ccx q[23], q[61], q[62];
ccx q[22], q[60], q[61];
ccx q[21], q[59], q[60];
ccx q[20], q[58], q[59];
ccx q[19], q[57], q[58];
ccx q[18], q[56], q[57];
ccx q[17], q[55], q[56];
ccx q[16], q[54], q[55];
ccx q[15], q[53], q[54];
ccx q[14], q[52], q[53];
ccx q[13], q[51], q[52];
ccx q[12], q[50], q[51];
ccx q[11], q[49], q[50];
ccx q[10], q[48], q[49];
ccx q[9], q[47], q[48];
ccx q[8], q[46], q[47];
ccx q[7], q[45], q[46];
ccx q[6], q[44], q[45];
ccx q[5], q[43], q[44];
ccx q[4], q[42], q[43];
ccx q[3], q[41], q[42];
ccx q[2], q[40], q[41];
ccx q[0], q[1], q[40];
ccx q[2], q[40], q[41];
ccx q[3], q[41], q[42];
ccx q[4], q[42], q[43];
ccx q[5], q[43], q[44];
ccx q[6], q[44], q[45];
ccx q[7], q[45], q[46];
ccx q[8], q[46], q[47];
ccx q[9], q[47], q[48];
ccx q[10], q[48], q[49];
ccx q[11], q[49], q[50];
ccx q[12], q[50], q[51];
ccx q[13], q[51], q[52];
ccx q[14], q[52], q[53];
ccx q[15], q[53], q[54];
ccx q[16], q[54], q[55];
ccx q[17], q[55], q[56];
ccx q[18], q[56], q[57];
ccx q[19], q[57], q[58];
ccx q[20], q[58], q[59];
ccx q[21], q[59], q[60];
ccx q[22], q[60], q[61];
ccx q[23], q[61], q[62];
ccx q[24], q[62], q[63];
ccx q[25], q[63], q[64];
ccx q[26], q[64], q[65];
ccx q[27], q[65], q[66];
ccx q[28], q[66], q[67];
ccx q[29], q[67], q[68];
ccx q[30], q[68], q[69];
ccx q[31], q[69], q[70];
ccx q[32], q[70], q[71];
ccx q[33], q[71], q[72];
ccx q[34], q[72], q[73];
ccx q[35], q[73], q[74];
ccx q[36], q[74], q[75];
ccx q[37], q[75], q[76];
ccx q[38], q[76], q[77];
ccx q[37], q[75], q[76];
ccx q[36], q[74], q[75];
ccx q[35], q[73], q[74];
ccx q[34], q[72], q[73];
ccx q[33], q[71], q[72];
ccx q[32], q[70], q[71];
ccx q[31], q[69], q[70];
ccx q[30], q[68], q[69];
ccx q[29], q[67], q[68];
ccx q[28], q[66], q[67];
ccx q[27], q[65], q[66];
ccx q[26], q[64], q[65];
ccx q[25], q[63], q[64];
ccx q[24], q[62], q[63];
ccx q[23], q[61], q[62];
ccx q[22], q[60], q[61];
ccx q[21], q[59], q[60];
ccx q[20], q[58], q[59];
ccx q[19], q[57], q[58];
ccx q[18], q[56], q[57];
ccx q[17], q[55], q[56];
ccx q[16], q[54], q[55];
ccx q[15], q[53], q[54];
ccx q[14], q[52], q[53];
ccx q[13], q[51], q[52];
ccx q[12], q[50], q[51];
ccx q[11], q[49], q[50];
ccx q[10], q[48], q[49];
ccx q[9], q[47], q[48];
ccx q[8], q[46], q[47];
ccx q[7], q[45], q[46];
ccx q[6], q[44], q[45];
ccx q[5], q[43], q[44];
ccx q[4], q[42], q[43];
ccx q[3], q[41], q[42];
ccx q[2], q[40], q[41];
ccx q[0], q[1], q[40];
ccx q[2], q[40], q[41];
ccx q[3], q[41], q[42];
ccx q[4], q[42], q[43];
ccx q[5], q[43], q[44];
ccx q[6], q[44], q[45];
ccx q[7], q[45], q[46];
ccx q[8], q[46], q[47];
ccx q[9], q[47], q[48];
ccx q[10], q[48], q[49];
ccx q[11], q[49], q[50];
ccx q[12], q[50], q[51];
ccx q[13], q[51], q[52];
ccx q[14], q[52], q[53];
ccx q[15], q[53], q[54];
ccx q[16], q[54], q[55];
ccx q[17], q[55], q[56];
ccx q[18], q[56], q[57];
ccx q[19], q[57], q[58];
ccx q[20], q[58], q[59];
ccx q[21], q[59], q[60];
ccx q[22], q[60], q[61];
ccx q[23], q[61], q[62];
ccx q[24], q[62], q[63];
ccx q[25], q[63], q[64];
ccx q[26], q[64], q[65];
ccx q[27], q[65], q[66];
ccx q[28], q[66], q[67];
ccx q[29], q[67], q[68];
ccx q[30], q[68], q[69];
ccx q[31], q[69], q[70];
ccx q[32], q[70], q[71];
ccx q[33], q[71], q[72];
ccx q[34], q[72], q[73];
ccx q[35], q[73], q[74];
ccx q[36], q[74], q[75];
ccx q[37], q[75], q[76];
h q[76];
s q[77];
ccx q[76], q[38], q[77];
ccx q[75], q[37], q[38];
ccx q[74], q[36], q[37];
ccx q[73], q[35], q[36];
ccx q[72], q[34], q[35];
ccx q[71], q[33], q[34];
ccx q[70], q[32], q[33];
ccx q[69], q[31], q[32];
ccx q[68], q[30], q[31];
ccx q[67], q[29], q[30];
ccx q[66], q[28], q[29];
ccx q[65], q[27], q[28];
ccx q[64], q[26], q[27];
ccx q[63], q[25], q[26];
ccx q[62], q[24], q[25];
ccx q[61], q[23], q[24];
ccx q[60], q[22], q[23];
ccx q[59], q[21], q[22];
ccx q[58], q[20], q[21];
ccx q[57], q[19], q[20];
ccx q[56], q[18], q[19];
ccx q[55], q[17], q[18];
ccx q[54], q[16], q[17];
ccx q[53], q[15], q[16];
ccx q[52], q[14], q[15];
ccx q[51], q[13], q[14];
ccx q[50], q[12], q[13];
ccx q[49], q[11], q[12];
ccx q[48], q[10], q[11];
ccx q[47], q[9], q[10];
ccx q[46], q[8], q[9];
ccx q[45], q[7], q[8];
ccx q[44], q[6], q[7];
ccx q[43], q[5], q[6];
ccx q[42], q[4], q[5];
ccx q[41], q[3], q[4];
ccx q[39], q[40], q[3];
ccx q[41], q[3], q[4];
ccx q[42], q[4], q[5];
ccx q[43], q[5], q[6];
ccx q[44], q[6], q[7];
ccx q[45], q[7], q[8];
ccx q[46], q[8], q[9];
ccx q[47], q[9], q[10];
ccx q[48], q[10], q[11];
ccx q[49], q[11], q[12];
ccx q[50], q[12], q[13];
ccx q[51], q[13], q[14];
ccx q[52], q[14], q[15];
ccx q[53], q[15], q[16];
ccx q[54], q[16], q[17];
ccx q[55], q[17], q[18];
ccx q[56], q[18], q[19];
ccx q[57], q[19], q[20];
ccx q[58], q[20], q[21];
ccx q[59], q[21], q[22];
ccx q[60], q[22], q[23];
ccx q[61], q[23], q[24];
ccx q[62], q[24], q[25];
ccx q[63], q[25], q[26];
ccx q[64], q[26], q[27];
ccx q[65], q[27], q[28];
ccx q[66], q[28], q[29];
ccx q[67], q[29], q[30];
ccx q[68], q[30], q[31];
ccx q[69], q[31], q[32];
ccx q[70], q[32], q[33];
ccx q[71], q[33], q[34];
ccx q[72], q[34], q[35];
ccx q[73], q[35], q[36];
ccx q[74], q[36], q[37];
ccx q[75], q[37], q[38];
ccx q[76], q[38], q[77];
ccx q[75], q[37], q[38];
ccx q[74], q[36], q[37];
ccx q[73], q[35], q[36];
ccx q[72], q[34], q[35];
ccx q[71], q[33], q[34];
ccx q[70], q[32], q[33];
ccx q[69], q[31], q[32];
ccx q[68], q[30], q[31];
ccx q[67], q[29], q[30];
ccx q[66], q[28], q[29];
ccx q[65], q[27], q[28];
ccx q[64], q[26], q[27];
ccx q[63], q[25], q[26];
ccx q[62], q[24], q[25];
ccx q[61], q[23], q[24];
ccx q[60], q[22], q[23];
ccx q[59], q[21], q[22];
ccx q[58], q[20], q[21];
ccx q[57], q[19], q[20];
ccx q[56], q[18], q[19];
ccx q[55], q[17], q[18];
ccx q[54], q[16], q[17];
ccx q[53], q[15], q[16];
ccx q[52], q[14], q[15];
ccx q[51], q[13], q[14];
ccx q[50], q[12], q[13];
ccx q[49], q[11], q[12];
ccx q[48], q[10], q[11];
ccx q[47], q[9], q[10];
ccx q[46], q[8], q[9];
ccx q[45], q[7], q[8];
ccx q[44], q[6], q[7];
ccx q[43], q[5], q[6];
ccx q[42], q[4], q[5];
ccx q[41], q[3], q[4];
ccx q[39], q[40], q[3];
ccx q[41], q[3], q[4];
ccx q[42], q[4], q[5];
ccx q[43], q[5], q[6];
ccx q[44], q[6], q[7];
ccx q[45], q[7], q[8];
ccx q[46], q[8], q[9];
ccx q[47], q[9], q[10];
ccx q[48], q[10], q[11];
ccx q[49], q[11], q[12];
ccx q[50], q[12], q[13];
ccx q[51], q[13], q[14];
ccx q[52], q[14], q[15];
ccx q[53], q[15], q[16];
ccx q[54], q[16], q[17];
ccx q[55], q[17], q[18];
ccx q[56], q[18], q[19];
ccx q[57], q[19], q[20];
ccx q[58], q[20], q[21];
ccx q[59], q[21], q[22];
ccx q[60], q[22], q[23];
ccx q[61], q[23], q[24];
ccx q[62], q[24], q[25];
ccx q[63], q[25], q[26];
ccx q[64], q[26], q[27];
ccx q[65], q[27], q[28];
ccx q[66], q[28], q[29];
ccx q[67], q[29], q[30];
ccx q[68], q[30], q[31];
ccx q[69], q[31], q[32];
ccx q[70], q[32], q[33];
ccx q[71], q[33], q[34];
ccx q[72], q[34], q[35];
ccx q[73], q[35], q[36];
ccx q[74], q[36], q[37];
ccx q[75], q[37], q[38];
sdg q[77];
ccx q[38], q[76], q[77];
ccx q[37], q[75], q[76];
ccx q[36], q[74], q[75];
ccx q[35], q[73], q[74];
ccx q[34], q[72], q[73];
ccx q[33], q[71], q[72];
ccx q[32], q[70], q[71];
ccx q[31], q[69], q[70];
ccx q[30], q[68], q[69];
ccx q[29], q[67], q[68];
ccx q[28], q[66], q[67];
ccx q[27], q[65], q[66];
ccx q[26], q[64], q[65];
ccx q[25], q[63], q[64];
ccx q[24], q[62], q[63];
ccx q[23], q[61], q[62];
ccx q[22], q[60], q[61];
ccx q[21], q[59], q[60];
ccx q[20], q[58], q[59];
ccx q[19], q[57], q[58];
ccx q[18], q[56], q[57];
ccx q[17], q[55], q[56];
ccx q[16], q[54], q[55];
ccx q[15], q[53], q[54];
ccx q[14], q[52], q[53];
ccx q[13], q[51], q[52];
ccx q[12], q[50], q[51];
ccx q[11], q[49], q[50];
ccx q[10], q[48], q[49];
ccx q[9], q[47], q[48];
ccx q[8], q[46], q[47];
ccx q[7], q[45], q[46];
ccx q[6], q[44], q[45];
ccx q[5], q[43], q[44];
ccx q[4], q[42], q[43];
ccx q[3], q[41], q[42];
ccx q[2], q[40], q[41];
ccx q[0], q[1], q[40];
ccx q[2], q[40], q[41];
ccx q[3], q[41], q[42];
ccx q[4], q[42], q[43];
ccx q[5], q[43], q[44];
ccx q[6], q[44], q[45];
ccx q[7], q[45], q[46];
ccx q[8], q[46], q[47];
ccx q[9], q[47], q[48];
ccx q[10], q[48], q[49];
ccx q[11], q[49], q[50];
ccx q[12], q[50], q[51];
ccx q[13], q[51], q[52];
ccx q[14], q[52], q[53];
ccx q[15], q[53], q[54];
ccx q[16], q[54], q[55];
ccx q[17], q[55], q[56];
ccx q[18], q[56], q[57];
ccx q[19], q[57], q[58];
ccx q[20], q[58], q[59];
ccx q[21], q[59], q[60];
ccx q[22], q[60], q[61];
ccx q[23], q[61], q[62];
ccx q[24], q[62], q[63];
ccx q[25], q[63], q[64];
ccx q[26], q[64], q[65];
ccx q[27], q[65], q[66];
ccx q[28], q[66], q[67];
ccx q[29], q[67], q[68];
ccx q[30], q[68], q[69];
ccx q[31], q[69], q[70];
ccx q[32], q[70], q[71];
ccx q[33], q[71], q[72];
ccx q[34], q[72], q[73];
ccx q[35], q[73], q[74];
ccx q[36], q[74], q[75];
ccx q[37], q[75], q[76];
ccx q[38], q[76], q[77];
ccx q[37], q[75], q[76];
ccx q[36], q[74], q[75];
ccx q[35], q[73], q[74];
ccx q[34], q[72], q[73];
ccx q[33], q[71], q[72];
ccx q[32], q[70], q[71];
ccx q[31], q[69], q[70];
ccx q[30], q[68], q[69];
ccx q[29], q[67], q[68];
ccx q[28], q[66], q[67];
ccx q[27], q[65], q[66];
ccx q[26], q[64], q[65];
ccx q[25], q[63], q[64];
ccx q[24], q[62], q[63];
ccx q[23], q[61], q[62];
ccx q[22], q[60], q[61];
ccx q[21], q[59], q[60];
ccx q[20], q[58], q[59];
ccx q[19], q[57], q[58];
ccx q[18], q[56], q[57];
ccx q[17], q[55], q[56];
ccx q[16], q[54], q[55];
ccx q[15], q[53], q[54];
ccx q[14], q[52], q[53];
ccx q[13], q[51], q[52];
ccx q[12], q[50], q[51];
ccx q[11], q[49], q[50];
ccx q[10], q[48], q[49];
ccx q[9], q[47], q[48];
ccx q[8], q[46], q[47];
ccx q[7], q[45], q[46];
ccx q[6], q[44], q[45];
ccx q[5], q[43], q[44];
ccx q[4], q[42], q[43];
ccx q[3], q[41], q[42];
ccx q[2], q[40], q[41];
ccx q[0], q[1], q[40];
ccx q[2], q[40], q[41];
ccx q[3], q[41], q[42];
ccx q[4], q[42], q[43];
ccx q[5], q[43], q[44];
ccx q[6], q[44], q[45];
ccx q[7], q[45], q[46];
ccx q[8], q[46], q[47];
ccx q[9], q[47], q[48];
ccx q[10], q[48], q[49];
ccx q[11], q[49], q[50];
ccx q[12], q[50], q[51];
ccx q[13], q[51], q[52];
ccx q[14], q[52], q[53];
ccx q[15], q[53], q[54];
ccx q[16], q[54], q[55];
ccx q[17], q[55], q[56];
ccx q[18], q[56], q[57];
ccx q[19], q[57], q[58];
ccx q[20], q[58], q[59];
ccx q[21], q[59], q[60];
ccx q[22], q[60], q[61];
ccx q[23], q[61], q[62];
ccx q[24], q[62], q[63];
ccx q[25], q[63], q[64];
ccx q[26], q[64], q[65];
ccx q[27], q[65], q[66];
ccx q[28], q[66], q[67];
ccx q[29], q[67], q[68];
ccx q[30], q[68], q[69];
ccx q[31], q[69], q[70];
ccx q[32], q[70], q[71];
ccx q[33], q[71], q[72];
ccx q[34], q[72], q[73];
ccx q[35], q[73], q[74];
ccx q[36], q[74], q[75];
ccx q[37], q[75], q[76];
s q[77];
ccx q[76], q[38], q[77];
ccx q[75], q[37], q[38];
ccx q[74], q[36], q[37];
ccx q[73], q[35], q[36];
ccx q[72], q[34], q[35];
ccx q[71], q[33], q[34];
ccx q[70], q[32], q[33];
ccx q[69], q[31], q[32];
ccx q[68], q[30], q[31];
ccx q[67], q[29], q[30];
ccx q[66], q[28], q[29];
ccx q[65], q[27], q[28];
ccx q[64], q[26], q[27];
ccx q[63], q[25], q[26];
ccx q[62], q[24], q[25];
ccx q[61], q[23], q[24];
ccx q[60], q[22], q[23];
ccx q[59], q[21], q[22];
ccx q[58], q[20], q[21];
ccx q[57], q[19], q[20];
ccx q[56], q[18], q[19];
ccx q[55], q[17], q[18];
ccx q[54], q[16], q[17];
ccx q[53], q[15], q[16];
ccx q[52], q[14], q[15];
ccx q[51], q[13], q[14];
ccx q[50], q[12], q[13];
ccx q[49], q[11], q[12];
ccx q[48], q[10], q[11];
ccx q[47], q[9], q[10];
ccx q[46], q[8], q[9];
ccx q[45], q[7], q[8];
ccx q[44], q[6], q[7];
ccx q[43], q[5], q[6];
ccx q[42], q[4], q[5];
ccx q[41], q[3], q[4];
ccx q[39], q[40], q[3];
ccx q[41], q[3], q[4];
ccx q[42], q[4], q[5];
ccx q[43], q[5], q[6];
ccx q[44], q[6], q[7];
ccx q[45], q[7], q[8];
ccx q[46], q[8], q[9];
ccx q[47], q[9], q[10];
ccx q[48], q[10], q[11];
ccx q[49], q[11], q[12];
ccx q[50], q[12], q[13];
ccx q[51], q[13], q[14];
ccx q[52], q[14], q[15];
ccx q[53], q[15], q[16];
ccx q[54], q[16], q[17];
ccx q[55], q[17], q[18];
ccx q[56], q[18], q[19];
ccx q[57], q[19], q[20];
ccx q[58], q[20], q[21];
ccx q[59], q[21], q[22];
ccx q[60], q[22], q[23];
ccx q[61], q[23], q[24];
ccx q[62], q[24], q[25];
ccx q[63], q[25], q[26];
ccx q[64], q[26], q[27];
ccx q[65], q[27], q[28];
ccx q[66], q[28], q[29];
ccx q[67], q[29], q[30];
ccx q[68], q[30], q[31];
ccx q[69], q[31], q[32];
ccx q[70], q[32], q[33];
ccx q[71], q[33], q[34];
ccx q[72], q[34], q[35];
ccx q[73], q[35], q[36];
ccx q[74], q[36], q[37];
ccx q[75], q[37], q[38];
ccx q[76], q[38], q[77];
ccx q[75], q[37], q[38];
ccx q[74], q[36], q[37];
ccx q[73], q[35], q[36];
ccx q[72], q[34], q[35];
ccx q[71], q[33], q[34];
ccx q[70], q[32], q[33];
ccx q[69], q[31], q[32];
ccx q[68], q[30], q[31];
ccx q[67], q[29], q[30];
ccx q[66], q[28], q[29];
ccx q[65], q[27], q[28];
ccx q[64], q[26], q[27];
ccx q[63], q[25], q[26];
ccx q[62], q[24], q[25];
ccx q[61], q[23], q[24];
ccx q[60], q[22], q[23];
ccx q[59], q[21], q[22];
ccx q[58], q[20], q[21];
ccx q[57], q[19], q[20];
ccx q[56], q[18], q[19];
ccx q[55], q[17], q[18];
ccx q[54], q[16], q[17];
ccx q[53], q[15], q[16];
ccx q[52], q[14], q[15];
ccx q[51], q[13], q[14];
ccx q[50], q[12], q[13];
ccx q[49], q[11], q[12];
ccx q[48], q[10], q[11];
ccx q[47], q[9], q[10];
ccx q[46], q[8], q[9];
ccx q[45], q[7], q[8];
ccx q[44], q[6], q[7];
ccx q[43], q[5], q[6];
ccx q[42], q[4], q[5];
ccx q[41], q[3], q[4];
ccx q[39], q[40], q[3];
ccx q[41], q[3], q[4];
ccx q[42], q[4], q[5];
ccx q[43], q[5], q[6];
ccx q[44], q[6], q[7];
ccx q[45], q[7], q[8];
ccx q[46], q[8], q[9];
ccx q[47], q[9], q[10];
ccx q[48], q[10], q[11];
ccx q[49], q[11], q[12];
ccx q[50], q[12], q[13];
ccx q[51], q[13], q[14];
ccx q[52], q[14], q[15];
ccx q[53], q[15], q[16];
ccx q[54], q[16], q[17];
ccx q[55], q[17], q[18];
ccx q[56], q[18], q[19];
ccx q[57], q[19], q[20];
ccx q[58], q[20], q[21];
ccx q[59], q[21], q[22];
ccx q[60], q[22], q[23];
ccx q[61], q[23], q[24];
ccx q[62], q[24], q[25];
ccx q[63], q[25], q[26];
ccx q[64], q[26], q[27];
ccx q[65], q[27], q[28];
ccx q[66], q[28], q[29];
ccx q[67], q[29], q[30];
ccx q[68], q[30], q[31];
ccx q[69], q[31], q[32];
ccx q[70], q[32], q[33];
ccx q[71], q[33], q[34];
ccx q[72], q[34], q[35];
ccx q[73], q[35], q[36];
ccx q[74], q[36], q[37];
ccx q[75], q[37], q[38];
h q[76];
sdg q[77];
h q[76];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
x q[6];
x q[7];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[13];
x q[14];
x q[15];
x q[16];
x q[17];
x q[18];
x q[19];
x q[20];
x q[21];
x q[22];
x q[23];
x q[24];
x q[25];
x q[26];
x q[27];
x q[28];
x q[29];
x q[30];
x q[31];
x q[32];
x q[33];
x q[34];
x q[35];
x q[36];
x q[37];
x q[38];
x q[39];
x q[40];
x q[41];
x q[42];
x q[43];
x q[44];
x q[45];
x q[46];
x q[47];
x q[48];
x q[49];
x q[50];
x q[51];
x q[52];
x q[53];
x q[54];
x q[55];
x q[56];
x q[57];
x q[58];
x q[59];
x q[60];
x q[61];
x q[62];
x q[63];
x q[64];
x q[65];
x q[66];
x q[67];
x q[68];
x q[69];
x q[70];
x q[71];
x q[72];
x q[73];
x q[74];
x q[75];
x q[76];
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
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];
measure q[30] -> c[30];
measure q[31] -> c[31];
measure q[32] -> c[32];
measure q[33] -> c[33];
measure q[34] -> c[34];
measure q[35] -> c[35];
measure q[36] -> c[36];
measure q[37] -> c[37];
measure q[38] -> c[38];
measure q[39] -> c[39];
measure q[40] -> c[40];
measure q[41] -> c[41];
measure q[42] -> c[42];
measure q[43] -> c[43];
measure q[44] -> c[44];
measure q[45] -> c[45];
measure q[46] -> c[46];
measure q[47] -> c[47];
measure q[48] -> c[48];
measure q[49] -> c[49];
measure q[50] -> c[50];
measure q[51] -> c[51];
measure q[52] -> c[52];
measure q[53] -> c[53];
measure q[54] -> c[54];
measure q[55] -> c[55];
measure q[56] -> c[56];
measure q[57] -> c[57];
measure q[58] -> c[58];
measure q[59] -> c[59];
measure q[60] -> c[60];
measure q[61] -> c[61];
measure q[62] -> c[62];
measure q[63] -> c[63];
measure q[64] -> c[64];
measure q[65] -> c[65];
measure q[66] -> c[66];
measure q[67] -> c[67];
measure q[68] -> c[68];
measure q[69] -> c[69];
measure q[70] -> c[70];
measure q[71] -> c[71];
measure q[72] -> c[72];
measure q[73] -> c[73];
measure q[74] -> c[74];
measure q[75] -> c[75];
measure q[76] -> c[76];
