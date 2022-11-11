OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
cu1(1.5707963267948966) q[10], q[16];
crz(1.5707963267948966) q[26], q[23];
crz(1.5707963267948966) q[17], q[4];
cz q[3], q[16];
cz q[14], q[10];
cz q[5], q[13];
cz q[26], q[10];
crz(1.5707963267948966) q[21], q[24];
crz(1.5707963267948966) q[19], q[12];
crz(1.5707963267948966) q[24], q[23];
crz(1.5707963267948966) q[6], q[5];
crz(1.5707963267948966) q[13], q[10];
crz(1.5707963267948966) q[15], q[8];
cz q[7], q[26];
cz q[20], q[10];
cz q[24], q[23];
cz q[2], q[12];
crz(1.5707963267948966) q[9], q[0];
crz(1.5707963267948966) q[8], q[12];
crz(1.5707963267948966) q[1], q[27];
cz q[13], q[15];
cu1(1.5707963267948966) q[2], q[18];
cz q[4], q[11];
cz q[8], q[22];
crz(1.5707963267948966) q[8], q[4];
crz(1.5707963267948966) q[26], q[21];
cu1(1.5707963267948966) q[22], q[4];
crz(1.5707963267948966) q[19], q[11];
cz q[7], q[22];
crz(1.5707963267948966) q[24], q[4];
cz q[9], q[12];
cu1(1.5707963267948966) q[17], q[19];
crz(1.5707963267948966) q[14], q[23];
cu1(1.5707963267948966) q[24], q[23];
cu1(1.5707963267948966) q[5], q[16];
cu1(1.5707963267948966) q[17], q[7];
crz(1.5707963267948966) q[5], q[16];
crz(1.5707963267948966) q[3], q[21];
crz(1.5707963267948966) q[12], q[6];
cz q[25], q[11];
crz(1.5707963267948966) q[8], q[11];
crz(1.5707963267948966) q[20], q[14];
crz(1.5707963267948966) q[10], q[4];
crz(1.5707963267948966) q[18], q[9];
cz q[16], q[13];
cz q[7], q[0];
cz q[9], q[24];
crz(1.5707963267948966) q[1], q[9];
cz q[16], q[17];
cu1(1.5707963267948966) q[19], q[6];
cz q[12], q[18];
crz(1.5707963267948966) q[26], q[23];
crz(1.5707963267948966) q[13], q[11];
crz(1.5707963267948966) q[16], q[19];
crz(1.5707963267948966) q[0], q[15];
cu1(1.5707963267948966) q[6], q[0];
cu1(1.5707963267948966) q[9], q[12];
crz(1.5707963267948966) q[5], q[20];
cz q[10], q[9];
cz q[5], q[21];
crz(1.5707963267948966) q[6], q[15];
cz q[7], q[12];
cu1(1.5707963267948966) q[24], q[0];
cu1(1.5707963267948966) q[4], q[9];
crz(1.5707963267948966) q[18], q[25];
cu1(1.5707963267948966) q[27], q[0];
cz q[0], q[11];
cu1(1.5707963267948966) q[0], q[20];
cu1(1.5707963267948966) q[2], q[9];
crz(1.5707963267948966) q[7], q[13];
cu1(1.5707963267948966) q[8], q[23];
cu1(1.5707963267948966) q[16], q[21];
crz(1.5707963267948966) q[18], q[25];
crz(1.5707963267948966) q[13], q[23];
cz q[24], q[6];
cz q[19], q[22];
crz(1.5707963267948966) q[14], q[18];
cu1(1.5707963267948966) q[3], q[12];
crz(1.5707963267948966) q[27], q[11];
crz(1.5707963267948966) q[15], q[7];
cz q[5], q[7];
crz(1.5707963267948966) q[22], q[27];
crz(1.5707963267948966) q[5], q[21];
cu1(1.5707963267948966) q[25], q[8];
crz(1.5707963267948966) q[18], q[4];
cu1(1.5707963267948966) q[13], q[20];
cu1(1.5707963267948966) q[1], q[18];
crz(1.5707963267948966) q[22], q[17];
cz q[8], q[4];
cz q[5], q[26];
cz q[2], q[10];
cz q[2], q[11];
cu1(1.5707963267948966) q[8], q[10];
cu1(1.5707963267948966) q[23], q[22];
cz q[4], q[21];
cu1(1.5707963267948966) q[9], q[25];
cz q[11], q[22];
crz(1.5707963267948966) q[6], q[8];
cz q[23], q[27];
cu1(1.5707963267948966) q[2], q[10];
crz(1.5707963267948966) q[0], q[7];
cz q[12], q[24];
cz q[24], q[10];
cu1(1.5707963267948966) q[10], q[15];
cu1(1.5707963267948966) q[4], q[8];
cz q[26], q[16];
cz q[25], q[18];
crz(1.5707963267948966) q[26], q[1];
crz(1.5707963267948966) q[1], q[13];
crz(1.5707963267948966) q[20], q[14];
cz q[22], q[21];
crz(1.5707963267948966) q[8], q[1];
cz q[26], q[20];
cz q[22], q[27];
crz(1.5707963267948966) q[5], q[8];
cu1(1.5707963267948966) q[27], q[2];
crz(1.5707963267948966) q[13], q[16];
cz q[1], q[2];
crz(1.5707963267948966) q[15], q[1];
crz(1.5707963267948966) q[18], q[3];
crz(1.5707963267948966) q[11], q[20];
crz(1.5707963267948966) q[23], q[13];
crz(1.5707963267948966) q[8], q[14];
crz(1.5707963267948966) q[7], q[12];
crz(1.5707963267948966) q[0], q[23];
cz q[26], q[16];
crz(1.5707963267948966) q[17], q[26];
crz(1.5707963267948966) q[9], q[7];
cz q[21], q[1];
crz(1.5707963267948966) q[2], q[3];
cz q[9], q[10];
crz(1.5707963267948966) q[1], q[14];
crz(1.5707963267948966) q[1], q[16];
cu1(1.5707963267948966) q[26], q[4];
cz q[12], q[25];
cz q[2], q[25];
cz q[21], q[4];
crz(1.5707963267948966) q[15], q[26];
cu1(1.5707963267948966) q[8], q[12];
cu1(1.5707963267948966) q[0], q[27];
crz(1.5707963267948966) q[20], q[25];
cu1(1.5707963267948966) q[8], q[17];
cz q[11], q[12];
cu1(1.5707963267948966) q[14], q[24];
cu1(1.5707963267948966) q[19], q[26];
cz q[26], q[11];
cz q[21], q[27];
crz(1.5707963267948966) q[3], q[9];
crz(1.5707963267948966) q[13], q[11];
crz(1.5707963267948966) q[17], q[7];
cu1(1.5707963267948966) q[10], q[27];
crz(1.5707963267948966) q[10], q[13];
cz q[16], q[6];
crz(1.5707963267948966) q[25], q[12];
crz(1.5707963267948966) q[22], q[17];
cz q[3], q[26];
crz(1.5707963267948966) q[9], q[14];
cu1(1.5707963267948966) q[13], q[19];
cu1(1.5707963267948966) q[7], q[10];
crz(1.5707963267948966) q[9], q[12];
crz(1.5707963267948966) q[15], q[25];
crz(1.5707963267948966) q[5], q[17];
cz q[17], q[24];
crz(1.5707963267948966) q[4], q[21];
cu1(1.5707963267948966) q[18], q[14];
cz q[19], q[18];
crz(1.5707963267948966) q[6], q[3];
cu1(1.5707963267948966) q[20], q[23];
crz(1.5707963267948966) q[27], q[13];
crz(1.5707963267948966) q[16], q[23];
crz(1.5707963267948966) q[26], q[13];
cz q[23], q[3];
cz q[24], q[14];
crz(1.5707963267948966) q[20], q[1];
cz q[19], q[9];
cu1(1.5707963267948966) q[26], q[24];
cz q[10], q[3];
cu1(1.5707963267948966) q[6], q[8];
crz(1.5707963267948966) q[1], q[22];
cu1(1.5707963267948966) q[16], q[7];
cu1(1.5707963267948966) q[7], q[1];
crz(1.5707963267948966) q[22], q[15];
crz(1.5707963267948966) q[8], q[23];
cz q[19], q[5];
cu1(1.5707963267948966) q[18], q[25];
cz q[4], q[23];
crz(1.5707963267948966) q[22], q[12];
crz(1.5707963267948966) q[1], q[0];
cu1(1.5707963267948966) q[16], q[26];
cu1(1.5707963267948966) q[3], q[14];
crz(1.5707963267948966) q[11], q[23];
cu1(1.5707963267948966) q[20], q[6];
crz(1.5707963267948966) q[4], q[24];
crz(1.5707963267948966) q[11], q[12];
cz q[2], q[16];
crz(1.5707963267948966) q[6], q[20];
