OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
creg c[27];
cu1(1.5707963267948966) q[5], q[12];
cz q[14], q[7];
crz(1.5707963267948966) q[0], q[5];
crz(1.5707963267948966) q[0], q[8];
cz q[20], q[11];
cz q[16], q[11];
cu1(1.5707963267948966) q[23], q[16];
crz(1.5707963267948966) q[0], q[21];
cu1(1.5707963267948966) q[7], q[24];
cz q[23], q[14];
cu1(1.5707963267948966) q[0], q[1];
crz(1.5707963267948966) q[6], q[14];
crz(1.5707963267948966) q[6], q[19];
cu1(1.5707963267948966) q[15], q[14];
crz(1.5707963267948966) q[4], q[7];
cu1(1.5707963267948966) q[17], q[15];
cu1(1.5707963267948966) q[22], q[11];
crz(1.5707963267948966) q[23], q[24];
cz q[6], q[13];
cz q[18], q[9];
cz q[11], q[2];
cu1(1.5707963267948966) q[17], q[4];
cu1(1.5707963267948966) q[26], q[3];
cz q[26], q[5];
cz q[7], q[24];
cz q[24], q[1];
cz q[0], q[21];
crz(1.5707963267948966) q[16], q[0];
cz q[26], q[17];
cu1(1.5707963267948966) q[3], q[11];
cz q[25], q[19];
cu1(1.5707963267948966) q[22], q[16];
crz(1.5707963267948966) q[1], q[9];
cu1(1.5707963267948966) q[0], q[3];
cz q[7], q[22];
cu1(1.5707963267948966) q[8], q[16];
crz(1.5707963267948966) q[22], q[17];
cz q[4], q[12];
crz(1.5707963267948966) q[23], q[18];
cz q[12], q[22];
crz(1.5707963267948966) q[9], q[16];
cz q[10], q[0];
cz q[23], q[19];
crz(1.5707963267948966) q[0], q[13];
cu1(1.5707963267948966) q[21], q[12];
cu1(1.5707963267948966) q[14], q[1];
crz(1.5707963267948966) q[24], q[23];
cz q[4], q[23];
crz(1.5707963267948966) q[7], q[12];
cz q[19], q[10];
cu1(1.5707963267948966) q[10], q[9];
crz(1.5707963267948966) q[24], q[21];
cz q[4], q[7];
cz q[2], q[15];
cz q[5], q[4];
cz q[21], q[0];
crz(1.5707963267948966) q[9], q[13];
crz(1.5707963267948966) q[12], q[3];
crz(1.5707963267948966) q[24], q[11];
crz(1.5707963267948966) q[26], q[23];
cz q[11], q[25];
cz q[20], q[7];
cz q[11], q[8];
cu1(1.5707963267948966) q[8], q[3];
crz(1.5707963267948966) q[9], q[23];
crz(1.5707963267948966) q[12], q[8];
cu1(1.5707963267948966) q[8], q[19];
cz q[4], q[1];
cz q[23], q[3];
cz q[17], q[2];
cz q[15], q[13];
cz q[5], q[13];
cu1(1.5707963267948966) q[2], q[23];
cu1(1.5707963267948966) q[25], q[19];
cu1(1.5707963267948966) q[21], q[5];
cu1(1.5707963267948966) q[20], q[16];
crz(1.5707963267948966) q[22], q[9];
cu1(1.5707963267948966) q[8], q[21];
crz(1.5707963267948966) q[21], q[26];
crz(1.5707963267948966) q[3], q[8];
crz(1.5707963267948966) q[6], q[15];
cz q[24], q[21];
cz q[15], q[19];
crz(1.5707963267948966) q[19], q[7];
crz(1.5707963267948966) q[22], q[4];
crz(1.5707963267948966) q[21], q[23];
crz(1.5707963267948966) q[0], q[25];
crz(1.5707963267948966) q[16], q[10];
crz(1.5707963267948966) q[16], q[7];
cu1(1.5707963267948966) q[11], q[10];
crz(1.5707963267948966) q[3], q[1];
cz q[14], q[6];
cz q[25], q[21];
crz(1.5707963267948966) q[9], q[14];
cz q[7], q[17];
cu1(1.5707963267948966) q[9], q[22];
cz q[2], q[4];
cz q[12], q[11];
cu1(1.5707963267948966) q[20], q[7];
cz q[11], q[25];
crz(1.5707963267948966) q[10], q[7];
cu1(1.5707963267948966) q[2], q[18];
cz q[19], q[3];
cu1(1.5707963267948966) q[21], q[20];
cz q[9], q[6];
cz q[10], q[6];
crz(1.5707963267948966) q[19], q[11];
crz(1.5707963267948966) q[7], q[5];
cu1(1.5707963267948966) q[9], q[4];
cu1(1.5707963267948966) q[25], q[24];
crz(1.5707963267948966) q[14], q[23];
crz(1.5707963267948966) q[13], q[1];
crz(1.5707963267948966) q[19], q[12];
crz(1.5707963267948966) q[9], q[17];
cz q[19], q[7];
cu1(1.5707963267948966) q[5], q[21];
crz(1.5707963267948966) q[19], q[23];
crz(1.5707963267948966) q[3], q[15];
cz q[20], q[6];
crz(1.5707963267948966) q[25], q[21];
crz(1.5707963267948966) q[19], q[12];
cu1(1.5707963267948966) q[17], q[12];
cz q[18], q[3];
crz(1.5707963267948966) q[7], q[1];
cz q[23], q[26];
cu1(1.5707963267948966) q[13], q[1];
crz(1.5707963267948966) q[23], q[18];
cu1(1.5707963267948966) q[0], q[6];
cz q[2], q[25];
cz q[4], q[2];
crz(1.5707963267948966) q[25], q[2];
cz q[25], q[13];
cu1(1.5707963267948966) q[4], q[21];
cz q[23], q[13];
cz q[23], q[24];
crz(1.5707963267948966) q[1], q[13];
cz q[11], q[19];
crz(1.5707963267948966) q[6], q[0];
cu1(1.5707963267948966) q[7], q[6];
cz q[17], q[13];
crz(1.5707963267948966) q[26], q[2];
crz(1.5707963267948966) q[6], q[26];
cu1(1.5707963267948966) q[0], q[5];
crz(1.5707963267948966) q[5], q[0];
crz(1.5707963267948966) q[9], q[2];
cu1(1.5707963267948966) q[7], q[3];
cz q[16], q[5];
crz(1.5707963267948966) q[16], q[19];
cz q[25], q[10];
cz q[1], q[5];
cz q[3], q[23];
crz(1.5707963267948966) q[25], q[10];
cz q[20], q[21];
cz q[19], q[8];
crz(1.5707963267948966) q[16], q[26];
crz(1.5707963267948966) q[3], q[16];
cu1(1.5707963267948966) q[20], q[2];
crz(1.5707963267948966) q[24], q[9];
cu1(1.5707963267948966) q[4], q[11];
crz(1.5707963267948966) q[7], q[0];
cz q[3], q[4];
cz q[16], q[24];
cu1(1.5707963267948966) q[5], q[7];
cu1(1.5707963267948966) q[7], q[4];
crz(1.5707963267948966) q[3], q[20];
crz(1.5707963267948966) q[12], q[16];
cu1(1.5707963267948966) q[1], q[22];
crz(1.5707963267948966) q[11], q[4];
cu1(1.5707963267948966) q[7], q[3];
cz q[5], q[17];
cz q[13], q[19];
cz q[24], q[7];
cu1(1.5707963267948966) q[22], q[21];
crz(1.5707963267948966) q[3], q[7];
cz q[1], q[0];
cu1(1.5707963267948966) q[10], q[14];
cu1(1.5707963267948966) q[14], q[17];
crz(1.5707963267948966) q[3], q[8];
cz q[20], q[23];
crz(1.5707963267948966) q[26], q[13];
cu1(1.5707963267948966) q[2], q[0];
cz q[3], q[12];
cz q[23], q[11];
crz(1.5707963267948966) q[20], q[25];
crz(1.5707963267948966) q[14], q[12];
cz q[2], q[4];
cz q[25], q[0];
cu1(1.5707963267948966) q[10], q[8];
cz q[18], q[9];
cz q[22], q[6];
crz(1.5707963267948966) q[0], q[5];
cu1(1.5707963267948966) q[12], q[10];
crz(1.5707963267948966) q[6], q[20];
crz(1.5707963267948966) q[18], q[20];
cz q[25], q[11];
crz(1.5707963267948966) q[12], q[20];
cu1(1.5707963267948966) q[2], q[16];
cu1(1.5707963267948966) q[12], q[23];
crz(1.5707963267948966) q[13], q[11];
cu1(1.5707963267948966) q[24], q[16];
cz q[15], q[24];
crz(1.5707963267948966) q[3], q[16];
cu1(1.5707963267948966) q[8], q[17];
cu1(1.5707963267948966) q[5], q[0];
cu1(1.5707963267948966) q[11], q[5];
cz q[11], q[26];
cu1(1.5707963267948966) q[15], q[1];
cz q[10], q[14];
cu1(1.5707963267948966) q[24], q[7];
cz q[22], q[9];
crz(1.5707963267948966) q[26], q[11];
crz(1.5707963267948966) q[6], q[21];
cu1(1.5707963267948966) q[7], q[4];
cz q[11], q[25];
crz(1.5707963267948966) q[26], q[2];
crz(1.5707963267948966) q[12], q[9];
cz q[2], q[16];
cz q[3], q[1];
cu1(1.5707963267948966) q[21], q[23];
cu1(1.5707963267948966) q[10], q[20];
cz q[6], q[20];
cu1(1.5707963267948966) q[4], q[24];
cz q[15], q[8];
crz(1.5707963267948966) q[15], q[24];
cz q[22], q[0];
crz(1.5707963267948966) q[11], q[22];
cz q[24], q[7];
cu1(1.5707963267948966) q[8], q[10];
crz(1.5707963267948966) q[10], q[3];
cz q[5], q[26];
cu1(1.5707963267948966) q[21], q[23];
cu1(1.5707963267948966) q[21], q[15];
crz(1.5707963267948966) q[4], q[23];
crz(1.5707963267948966) q[19], q[26];
cu1(1.5707963267948966) q[3], q[8];
cu1(1.5707963267948966) q[1], q[18];
crz(1.5707963267948966) q[14], q[7];
cz q[22], q[1];
cu1(1.5707963267948966) q[8], q[21];
cu1(1.5707963267948966) q[9], q[14];
crz(1.5707963267948966) q[10], q[8];
crz(1.5707963267948966) q[23], q[11];
cu1(1.5707963267948966) q[22], q[10];
cu1(1.5707963267948966) q[11], q[25];
crz(1.5707963267948966) q[16], q[13];
cz q[22], q[11];
cz q[23], q[3];
cu1(1.5707963267948966) q[9], q[0];
cz q[20], q[15];
crz(1.5707963267948966) q[12], q[15];
crz(1.5707963267948966) q[13], q[15];
cz q[11], q[1];
crz(1.5707963267948966) q[23], q[20];
cz q[15], q[24];
crz(1.5707963267948966) q[19], q[2];
cu1(1.5707963267948966) q[15], q[16];
crz(1.5707963267948966) q[15], q[11];
cu1(1.5707963267948966) q[10], q[4];
cz q[24], q[9];
cz q[22], q[18];
crz(1.5707963267948966) q[5], q[10];
cu1(1.5707963267948966) q[16], q[4];
cu1(1.5707963267948966) q[26], q[19];
cz q[4], q[18];
crz(1.5707963267948966) q[12], q[23];
cu1(1.5707963267948966) q[1], q[12];
cu1(1.5707963267948966) q[22], q[21];
cu1(1.5707963267948966) q[11], q[7];
cu1(1.5707963267948966) q[14], q[25];
cu1(1.5707963267948966) q[13], q[23];
crz(1.5707963267948966) q[14], q[24];
cz q[9], q[6];
cu1(1.5707963267948966) q[13], q[2];
cz q[19], q[10];
cz q[17], q[25];
cz q[13], q[22];
crz(1.5707963267948966) q[15], q[9];
cu1(1.5707963267948966) q[3], q[22];
cu1(1.5707963267948966) q[21], q[26];
cz q[16], q[7];
cu1(1.5707963267948966) q[4], q[15];
cu1(1.5707963267948966) q[15], q[22];
cz q[7], q[15];
cz q[2], q[26];
cu1(1.5707963267948966) q[24], q[19];
cu1(1.5707963267948966) q[16], q[2];
cu1(1.5707963267948966) q[0], q[20];
crz(1.5707963267948966) q[24], q[22];
cu1(1.5707963267948966) q[19], q[23];
crz(1.5707963267948966) q[5], q[18];
cz q[15], q[21];
cz q[9], q[12];
cu1(1.5707963267948966) q[14], q[0];
crz(1.5707963267948966) q[4], q[0];
cz q[8], q[3];
cu1(1.5707963267948966) q[9], q[1];
cu1(1.5707963267948966) q[19], q[25];
crz(1.5707963267948966) q[20], q[1];
cu1(1.5707963267948966) q[9], q[16];
crz(1.5707963267948966) q[9], q[15];
cz q[10], q[18];
crz(1.5707963267948966) q[16], q[5];
crz(1.5707963267948966) q[17], q[26];
crz(1.5707963267948966) q[19], q[0];
crz(1.5707963267948966) q[10], q[14];
cz q[17], q[22];
crz(1.5707963267948966) q[10], q[8];
cu1(1.5707963267948966) q[12], q[19];
cu1(1.5707963267948966) q[1], q[4];
cz q[3], q[18];
cu1(1.5707963267948966) q[26], q[25];
cu1(1.5707963267948966) q[26], q[7];
crz(1.5707963267948966) q[5], q[20];
cz q[3], q[0];
cu1(1.5707963267948966) q[22], q[20];
crz(1.5707963267948966) q[26], q[6];
cu1(1.5707963267948966) q[13], q[18];
cu1(1.5707963267948966) q[23], q[24];
crz(1.5707963267948966) q[14], q[26];
cz q[12], q[5];
crz(1.5707963267948966) q[18], q[23];
crz(1.5707963267948966) q[17], q[2];
cu1(1.5707963267948966) q[9], q[15];
crz(1.5707963267948966) q[10], q[26];
cz q[13], q[9];
crz(1.5707963267948966) q[1], q[12];
crz(1.5707963267948966) q[16], q[13];
crz(1.5707963267948966) q[0], q[4];
crz(1.5707963267948966) q[0], q[23];
cu1(1.5707963267948966) q[20], q[25];
cu1(1.5707963267948966) q[1], q[18];
cu1(1.5707963267948966) q[20], q[19];
cu1(1.5707963267948966) q[16], q[22];
crz(1.5707963267948966) q[10], q[12];
cz q[0], q[11];
cu1(1.5707963267948966) q[6], q[24];
crz(1.5707963267948966) q[0], q[3];
cu1(1.5707963267948966) q[6], q[15];
cu1(1.5707963267948966) q[5], q[14];
crz(1.5707963267948966) q[5], q[12];
crz(1.5707963267948966) q[11], q[20];
cz q[11], q[19];
crz(1.5707963267948966) q[4], q[26];
cu1(1.5707963267948966) q[18], q[23];
cz q[25], q[12];
cz q[23], q[26];
crz(1.5707963267948966) q[7], q[14];
cz q[26], q[12];
cu1(1.5707963267948966) q[13], q[12];
cu1(1.5707963267948966) q[8], q[26];
crz(1.5707963267948966) q[25], q[19];
cu1(1.5707963267948966) q[18], q[19];
cu1(1.5707963267948966) q[1], q[16];
crz(1.5707963267948966) q[9], q[19];
cz q[2], q[21];
cu1(1.5707963267948966) q[9], q[17];
cu1(1.5707963267948966) q[4], q[13];
cz q[13], q[4];
cz q[2], q[17];
cu1(1.5707963267948966) q[6], q[11];
crz(1.5707963267948966) q[11], q[18];
crz(1.5707963267948966) q[8], q[11];
crz(1.5707963267948966) q[7], q[19];
crz(1.5707963267948966) q[6], q[23];
cu1(1.5707963267948966) q[9], q[26];
crz(1.5707963267948966) q[24], q[6];
crz(1.5707963267948966) q[17], q[25];
crz(1.5707963267948966) q[17], q[22];
cz q[5], q[19];
crz(1.5707963267948966) q[10], q[9];
cu1(1.5707963267948966) q[3], q[9];
cz q[22], q[11];
cu1(1.5707963267948966) q[18], q[9];
crz(1.5707963267948966) q[25], q[10];
crz(1.5707963267948966) q[25], q[12];
cu1(1.5707963267948966) q[15], q[7];
cz q[4], q[5];
cz q[9], q[25];
cu1(1.5707963267948966) q[26], q[22];
crz(1.5707963267948966) q[6], q[15];
cz q[4], q[8];
cu1(1.5707963267948966) q[1], q[20];
cz q[1], q[11];
crz(1.5707963267948966) q[17], q[26];
cz q[24], q[26];
cz q[0], q[24];
cu1(1.5707963267948966) q[16], q[13];
cz q[18], q[23];
crz(1.5707963267948966) q[3], q[2];
cz q[5], q[23];
crz(1.5707963267948966) q[20], q[13];
cz q[16], q[18];
cz q[21], q[17];
crz(1.5707963267948966) q[15], q[12];
crz(1.5707963267948966) q[2], q[10];
cz q[1], q[5];
crz(1.5707963267948966) q[8], q[0];
crz(1.5707963267948966) q[22], q[24];
crz(1.5707963267948966) q[3], q[22];
cz q[19], q[7];
cu1(1.5707963267948966) q[2], q[10];
crz(1.5707963267948966) q[17], q[0];
crz(1.5707963267948966) q[6], q[16];
cu1(1.5707963267948966) q[7], q[12];
cu1(1.5707963267948966) q[11], q[22];
crz(1.5707963267948966) q[22], q[11];
crz(1.5707963267948966) q[4], q[12];
crz(1.5707963267948966) q[12], q[2];
crz(1.5707963267948966) q[2], q[23];
cz q[19], q[7];
cu1(1.5707963267948966) q[17], q[22];
crz(1.5707963267948966) q[8], q[22];
cz q[6], q[3];
cu1(1.5707963267948966) q[24], q[7];
crz(1.5707963267948966) q[8], q[25];
crz(1.5707963267948966) q[11], q[8];
crz(1.5707963267948966) q[2], q[19];
crz(1.5707963267948966) q[13], q[15];
crz(1.5707963267948966) q[26], q[25];
cz q[6], q[15];
cu1(1.5707963267948966) q[1], q[18];
cu1(1.5707963267948966) q[19], q[26];
cu1(1.5707963267948966) q[10], q[0];
cu1(1.5707963267948966) q[0], q[20];
cz q[3], q[9];
cz q[22], q[1];
cu1(1.5707963267948966) q[9], q[22];
cz q[24], q[0];
crz(1.5707963267948966) q[16], q[21];
crz(1.5707963267948966) q[14], q[15];
cz q[11], q[15];
cu1(1.5707963267948966) q[6], q[5];
crz(1.5707963267948966) q[26], q[11];
cz q[16], q[17];
cu1(1.5707963267948966) q[7], q[16];
crz(1.5707963267948966) q[8], q[9];
cu1(1.5707963267948966) q[23], q[25];
cz q[19], q[22];
cz q[7], q[5];
cu1(1.5707963267948966) q[21], q[4];
cu1(1.5707963267948966) q[14], q[18];
cu1(1.5707963267948966) q[24], q[19];
cu1(1.5707963267948966) q[16], q[25];
cz q[9], q[8];
crz(1.5707963267948966) q[25], q[12];
cu1(1.5707963267948966) q[21], q[19];
cu1(1.5707963267948966) q[0], q[25];
cz q[17], q[19];
crz(1.5707963267948966) q[10], q[14];
cu1(1.5707963267948966) q[7], q[4];
crz(1.5707963267948966) q[4], q[8];
cz q[16], q[8];
cu1(1.5707963267948966) q[6], q[19];
crz(1.5707963267948966) q[20], q[25];
cu1(1.5707963267948966) q[8], q[4];
cz q[0], q[20];
cu1(1.5707963267948966) q[20], q[0];
cu1(1.5707963267948966) q[8], q[6];
crz(1.5707963267948966) q[24], q[3];
cu1(1.5707963267948966) q[11], q[8];
cu1(1.5707963267948966) q[10], q[15];
cu1(1.5707963267948966) q[23], q[6];
cu1(1.5707963267948966) q[11], q[24];
cu1(1.5707963267948966) q[2], q[20];
cu1(1.5707963267948966) q[17], q[0];
crz(1.5707963267948966) q[23], q[1];
cu1(1.5707963267948966) q[9], q[23];
cz q[12], q[24];
cz q[15], q[14];
cu1(1.5707963267948966) q[14], q[3];
cz q[5], q[20];
cu1(1.5707963267948966) q[1], q[13];
cu1(1.5707963267948966) q[13], q[20];
cz q[22], q[12];
cu1(1.5707963267948966) q[1], q[3];
cz q[10], q[18];
cz q[2], q[21];
crz(1.5707963267948966) q[3], q[15];
crz(1.5707963267948966) q[5], q[13];
crz(1.5707963267948966) q[20], q[24];
cz q[13], q[14];
cz q[20], q[24];
crz(1.5707963267948966) q[13], q[4];
cu1(1.5707963267948966) q[7], q[10];
crz(1.5707963267948966) q[0], q[2];
cz q[12], q[3];
