OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
h q[3];
h q[14];
h q[17];
h q[25];
cx q[3], q[16];
h q[5];
cx q[2], q[4];
cx q[6], q[21];
cx q[23], q[27];
h q[27];
cx q[26], q[19];
h q[28];
cx q[16], q[8];
h q[27];
h q[23];
h q[26];
cx q[7], q[0];
h q[23];
h q[25];
cx q[12], q[28];
cx q[24], q[10];
cx q[1], q[24];
h q[3];
h q[22];
cx q[2], q[4];
cx q[7], q[26];
cx q[7], q[22];
h q[19];
h q[12];
h q[14];
h q[0];
h q[13];
h q[21];
cx q[18], q[4];
cx q[12], q[9];
h q[2];
h q[28];
h q[1];
cx q[8], q[25];
cx q[18], q[9];
h q[24];
h q[7];
h q[23];
cx q[26], q[15];
cx q[27], q[4];
cx q[4], q[23];
h q[12];
h q[2];
h q[25];
h q[27];
h q[21];
h q[16];
cx q[14], q[27];
h q[25];
h q[11];
h q[13];
h q[3];
h q[27];
h q[5];
h q[6];
h q[20];
h q[12];
cx q[10], q[26];
h q[7];
h q[10];
cx q[23], q[10];
cx q[26], q[18];
h q[9];
h q[7];
cx q[4], q[20];
h q[6];
cx q[1], q[7];
cx q[17], q[10];
h q[17];
cx q[16], q[8];
h q[8];
h q[1];
cx q[13], q[9];
h q[15];
h q[26];
cx q[16], q[8];
cx q[12], q[28];
h q[16];
h q[27];
cx q[15], q[9];
h q[28];
cx q[0], q[27];
cx q[5], q[12];
h q[19];
cx q[9], q[4];
h q[24];
cx q[1], q[24];
h q[23];
h q[14];
cx q[1], q[0];
cx q[17], q[26];
cx q[12], q[18];
cx q[21], q[23];
cx q[23], q[13];
h q[7];
cx q[16], q[9];
cx q[20], q[25];
h q[21];
h q[19];
h q[10];
h q[9];
cx q[15], q[18];
h q[23];
h q[20];
cx q[18], q[3];
h q[24];
cx q[6], q[22];
h q[5];
cx q[20], q[9];
h q[12];
cx q[3], q[25];
cx q[24], q[0];
cx q[19], q[24];
h q[12];
h q[0];
h q[2];
cx q[28], q[1];
h q[9];
cx q[0], q[25];
h q[20];
h q[3];
cx q[5], q[28];
cx q[3], q[19];
h q[7];
cx q[23], q[19];
h q[13];
cx q[4], q[15];
h q[18];
h q[0];
cx q[27], q[16];
h q[3];
cx q[0], q[7];
cx q[8], q[27];
cx q[2], q[26];
h q[2];
h q[5];
cx q[15], q[1];
cx q[16], q[3];
cx q[19], q[5];
h q[27];
cx q[26], q[19];
cx q[18], q[17];
cx q[13], q[26];
h q[2];
h q[13];
h q[0];
cx q[20], q[3];
cx q[1], q[24];
cx q[16], q[15];
cx q[26], q[13];
h q[26];
cx q[21], q[6];
cx q[1], q[10];
h q[12];
cx q[25], q[16];
h q[3];
h q[0];
cx q[0], q[15];
cx q[19], q[0];
h q[23];
cx q[28], q[6];
cx q[18], q[22];
cx q[4], q[21];
h q[26];
cx q[23], q[17];
cx q[2], q[18];
h q[6];
cx q[4], q[14];
h q[5];
h q[21];
cx q[24], q[5];
h q[16];
h q[18];
cx q[15], q[28];
h q[11];
cx q[14], q[21];
cx q[18], q[1];
h q[16];
h q[10];
cx q[7], q[22];
cx q[13], q[23];
h q[28];
h q[0];
h q[24];
h q[9];
h q[17];
h q[24];
cx q[3], q[27];
h q[1];
cx q[4], q[14];
cx q[21], q[15];
h q[12];
cx q[9], q[12];
h q[11];
h q[23];
h q[24];
h q[28];
cx q[24], q[4];
cx q[10], q[28];
cx q[2], q[9];
cx q[15], q[20];
h q[8];
h q[13];
cx q[0], q[2];
cx q[8], q[18];
h q[19];
cx q[25], q[10];
h q[10];
cx q[14], q[0];
cx q[3], q[5];
h q[15];
cx q[4], q[1];
h q[2];
cx q[20], q[23];
h q[10];
h q[27];
cx q[14], q[11];
h q[6];
cx q[21], q[5];
cx q[7], q[17];
h q[10];
cx q[1], q[10];
cx q[14], q[19];
cx q[21], q[9];
cx q[20], q[2];
cx q[19], q[13];
h q[5];
h q[24];
cx q[14], q[6];
cx q[25], q[0];
cx q[14], q[18];
cx q[12], q[1];
cx q[20], q[11];
cx q[5], q[7];
cx q[25], q[20];
h q[21];
h q[27];
h q[11];
h q[2];
cx q[13], q[12];
h q[12];
h q[5];
cx q[23], q[0];
h q[23];
cx q[10], q[7];
h q[27];
cx q[25], q[17];
cx q[6], q[25];
h q[3];
cx q[1], q[5];
cx q[5], q[0];
cx q[1], q[5];
h q[10];
h q[4];
cx q[11], q[16];
cx q[22], q[12];
h q[18];
cx q[3], q[1];
cx q[6], q[27];
cx q[15], q[2];
cx q[11], q[3];
h q[17];
h q[23];
cx q[13], q[28];
h q[23];
h q[3];
h q[2];
cx q[22], q[13];
h q[21];
cx q[4], q[15];
h q[28];
cx q[18], q[14];
h q[2];
cx q[11], q[9];
cx q[2], q[18];
h q[1];
h q[13];
h q[17];
h q[3];
h q[2];
cx q[3], q[21];
h q[15];
h q[15];
h q[16];
h q[4];
h q[25];
h q[2];
h q[17];
h q[19];
cx q[18], q[27];
cx q[22], q[14];
h q[23];
h q[22];
cx q[22], q[23];
cx q[2], q[22];
cx q[8], q[17];
cx q[20], q[16];
h q[27];
cx q[28], q[7];
cx q[23], q[25];
cx q[2], q[10];
h q[3];
h q[23];
cx q[20], q[1];
h q[12];
cx q[11], q[4];
h q[18];
cx q[10], q[8];
cx q[3], q[15];
cx q[18], q[26];
cx q[13], q[17];
cx q[1], q[22];
cx q[19], q[21];
h q[3];
cx q[27], q[0];
cx q[23], q[25];
h q[3];
h q[14];
cx q[25], q[23];
cx q[26], q[21];
h q[16];
cx q[7], q[2];
h q[11];
cx q[15], q[2];
h q[1];
cx q[12], q[10];
h q[13];
h q[2];
h q[19];
h q[19];
h q[16];
h q[3];
h q[13];
cx q[25], q[22];
h q[11];
cx q[19], q[21];
cx q[26], q[25];
cx q[8], q[24];
cx q[1], q[3];
h q[14];
cx q[20], q[10];
h q[23];
h q[12];
cx q[11], q[19];
cx q[16], q[19];
cx q[28], q[26];
h q[1];
cx q[2], q[23];
cx q[8], q[13];
cx q[4], q[28];
cx q[10], q[9];
cx q[24], q[15];
cx q[22], q[16];
cx q[19], q[0];
cx q[0], q[22];
cx q[24], q[22];
h q[18];
cx q[15], q[8];
h q[19];
cx q[12], q[7];
h q[6];
cx q[5], q[23];
h q[3];
h q[17];
h q[1];
h q[19];
cx q[25], q[11];
cx q[19], q[26];
h q[11];
h q[18];
h q[21];
h q[6];
h q[28];
cx q[12], q[2];
h q[17];
cx q[1], q[21];
h q[18];
cx q[17], q[9];
cx q[6], q[18];
h q[20];
cx q[1], q[28];
cx q[24], q[2];
h q[12];
cx q[24], q[27];
h q[25];
h q[2];
cx q[6], q[12];
h q[11];
h q[3];
h q[4];
cx q[20], q[16];
cx q[6], q[21];
h q[25];
h q[4];
cx q[26], q[19];
h q[27];
h q[22];
h q[19];
h q[4];
cx q[17], q[10];
cx q[11], q[15];
cx q[0], q[23];
h q[2];
h q[1];
cx q[9], q[6];
h q[23];
h q[18];
cx q[12], q[9];
cx q[18], q[10];
h q[21];
cx q[22], q[27];
cx q[8], q[0];
cx q[21], q[26];
cx q[20], q[7];
cx q[2], q[28];
cx q[17], q[20];
cx q[5], q[16];
h q[22];
cx q[13], q[20];
cx q[6], q[27];
cx q[6], q[3];
h q[7];
h q[24];
h q[11];
h q[27];
cx q[7], q[19];
h q[20];
cx q[18], q[10];
h q[18];
h q[19];
cx q[11], q[5];
h q[4];
cx q[18], q[26];
cx q[18], q[16];
cx q[3], q[10];
cx q[25], q[18];
cx q[8], q[0];
h q[26];
cx q[16], q[7];
h q[10];
h q[23];
cx q[23], q[10];
cx q[13], q[2];
cx q[10], q[15];
cx q[25], q[20];
h q[18];
cx q[5], q[14];
cx q[20], q[26];
h q[0];
h q[10];
h q[12];
h q[7];
cx q[8], q[11];
cx q[21], q[17];
cx q[12], q[7];
h q[19];
h q[14];
h q[15];
h q[26];
cx q[2], q[21];
cx q[26], q[16];
cx q[22], q[0];
h q[5];
h q[0];
cx q[14], q[1];
h q[9];
cx q[28], q[17];
cx q[11], q[22];
cx q[22], q[28];
h q[20];
h q[8];
cx q[27], q[23];
cx q[24], q[15];
h q[23];
h q[14];
h q[11];
h q[10];
cx q[6], q[20];
h q[3];
cx q[0], q[1];
cx q[10], q[24];
h q[5];
h q[23];
h q[14];
h q[6];
cx q[26], q[3];
cx q[24], q[6];
cx q[23], q[11];
cx q[24], q[16];
h q[16];
cx q[10], q[23];
h q[6];
h q[13];
h q[18];
h q[19];
cx q[1], q[21];
h q[4];
cx q[14], q[5];
h q[25];
h q[4];
cx q[12], q[10];
cx q[11], q[17];
h q[14];
h q[25];
h q[22];
cx q[12], q[11];
h q[23];
cx q[8], q[19];
cx q[23], q[22];
h q[7];
h q[11];
cx q[22], q[27];
cx q[28], q[8];
h q[23];
h q[3];
h q[3];
h q[24];
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
