OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
cz q[12], q[7];
cz q[26], q[14];
cu1(1.5707963267948966) q[15], q[21];
cu1(1.5707963267948966) q[18], q[28];
crz(1.5707963267948966) q[27], q[19];
crz(1.5707963267948966) q[29], q[28];
crz(1.5707963267948966) q[2], q[20];
cu1(1.5707963267948966) q[28], q[0];
cu1(1.5707963267948966) q[15], q[25];
cu1(1.5707963267948966) q[17], q[29];
cu1(1.5707963267948966) q[20], q[0];
cz q[26], q[17];
cu1(1.5707963267948966) q[12], q[11];
cu1(1.5707963267948966) q[19], q[21];
crz(1.5707963267948966) q[14], q[1];
cu1(1.5707963267948966) q[15], q[1];
crz(1.5707963267948966) q[12], q[15];
cz q[11], q[12];
cu1(1.5707963267948966) q[3], q[4];
cu1(1.5707963267948966) q[5], q[1];
cz q[17], q[2];
cu1(1.5707963267948966) q[23], q[18];
cu1(1.5707963267948966) q[8], q[12];
crz(1.5707963267948966) q[29], q[1];
crz(1.5707963267948966) q[11], q[28];
crz(1.5707963267948966) q[11], q[19];
crz(1.5707963267948966) q[22], q[28];
crz(1.5707963267948966) q[14], q[25];
cz q[27], q[7];
cu1(1.5707963267948966) q[2], q[7];
crz(1.5707963267948966) q[9], q[4];
cz q[19], q[16];
cz q[0], q[3];
cz q[29], q[8];
crz(1.5707963267948966) q[15], q[12];
crz(1.5707963267948966) q[6], q[10];
cu1(1.5707963267948966) q[5], q[16];
crz(1.5707963267948966) q[5], q[22];
cz q[4], q[25];
cu1(1.5707963267948966) q[17], q[28];
cu1(1.5707963267948966) q[15], q[10];
cz q[7], q[5];
cz q[6], q[14];
cu1(1.5707963267948966) q[23], q[3];
cu1(1.5707963267948966) q[25], q[16];
cu1(1.5707963267948966) q[10], q[2];
crz(1.5707963267948966) q[0], q[13];
cz q[6], q[28];
cz q[23], q[6];
cu1(1.5707963267948966) q[4], q[7];
cz q[9], q[15];
cu1(1.5707963267948966) q[13], q[21];
cu1(1.5707963267948966) q[16], q[23];
cz q[27], q[8];
cz q[13], q[4];
crz(1.5707963267948966) q[15], q[23];
crz(1.5707963267948966) q[14], q[5];
cu1(1.5707963267948966) q[10], q[14];
crz(1.5707963267948966) q[17], q[2];
cz q[10], q[13];
crz(1.5707963267948966) q[4], q[20];
cu1(1.5707963267948966) q[8], q[19];
cz q[20], q[27];
cu1(1.5707963267948966) q[13], q[16];
crz(1.5707963267948966) q[15], q[12];
cz q[11], q[8];
cz q[10], q[5];
crz(1.5707963267948966) q[6], q[27];
crz(1.5707963267948966) q[23], q[21];
cz q[1], q[13];
crz(1.5707963267948966) q[29], q[27];
cz q[20], q[7];
crz(1.5707963267948966) q[4], q[25];
cz q[14], q[2];
cz q[4], q[7];
cz q[9], q[10];
cu1(1.5707963267948966) q[18], q[13];
crz(1.5707963267948966) q[1], q[26];
crz(1.5707963267948966) q[16], q[24];
cz q[5], q[8];
crz(1.5707963267948966) q[4], q[13];
cu1(1.5707963267948966) q[17], q[6];
cu1(1.5707963267948966) q[15], q[10];
crz(1.5707963267948966) q[2], q[7];
crz(1.5707963267948966) q[14], q[9];
cz q[4], q[23];
cu1(1.5707963267948966) q[8], q[0];
crz(1.5707963267948966) q[0], q[7];
cu1(1.5707963267948966) q[22], q[8];
cz q[5], q[20];
crz(1.5707963267948966) q[24], q[15];
crz(1.5707963267948966) q[12], q[22];
cz q[25], q[12];
cu1(1.5707963267948966) q[3], q[18];
crz(1.5707963267948966) q[7], q[6];
crz(1.5707963267948966) q[18], q[2];
cu1(1.5707963267948966) q[16], q[1];
crz(1.5707963267948966) q[12], q[21];
crz(1.5707963267948966) q[17], q[23];
cu1(1.5707963267948966) q[11], q[3];
crz(1.5707963267948966) q[22], q[0];
crz(1.5707963267948966) q[13], q[10];
cz q[1], q[19];
crz(1.5707963267948966) q[8], q[24];
crz(1.5707963267948966) q[18], q[19];
crz(1.5707963267948966) q[17], q[16];
cu1(1.5707963267948966) q[7], q[16];
cz q[12], q[25];
cz q[27], q[19];
cu1(1.5707963267948966) q[26], q[1];
cz q[27], q[11];
cu1(1.5707963267948966) q[20], q[24];
crz(1.5707963267948966) q[1], q[2];
cz q[26], q[21];
crz(1.5707963267948966) q[11], q[8];
cz q[8], q[1];
cz q[7], q[13];
crz(1.5707963267948966) q[26], q[25];
cz q[20], q[17];
cu1(1.5707963267948966) q[21], q[20];
cu1(1.5707963267948966) q[28], q[6];
crz(1.5707963267948966) q[7], q[1];
crz(1.5707963267948966) q[2], q[18];
crz(1.5707963267948966) q[22], q[10];
crz(1.5707963267948966) q[28], q[4];
cz q[14], q[6];
crz(1.5707963267948966) q[17], q[22];
cu1(1.5707963267948966) q[19], q[5];
crz(1.5707963267948966) q[5], q[11];
cu1(1.5707963267948966) q[16], q[3];
cz q[22], q[23];
cu1(1.5707963267948966) q[11], q[24];
cz q[29], q[1];
cu1(1.5707963267948966) q[9], q[18];
cu1(1.5707963267948966) q[1], q[5];
crz(1.5707963267948966) q[7], q[21];
cz q[13], q[11];
cu1(1.5707963267948966) q[12], q[15];
cz q[13], q[19];
cu1(1.5707963267948966) q[0], q[19];
crz(1.5707963267948966) q[24], q[25];
cu1(1.5707963267948966) q[9], q[18];
crz(1.5707963267948966) q[11], q[10];
cu1(1.5707963267948966) q[11], q[23];
crz(1.5707963267948966) q[0], q[14];
cu1(1.5707963267948966) q[13], q[22];
cu1(1.5707963267948966) q[22], q[11];
cz q[14], q[2];
crz(1.5707963267948966) q[7], q[6];
cu1(1.5707963267948966) q[17], q[1];
crz(1.5707963267948966) q[2], q[4];
cz q[16], q[14];
cu1(1.5707963267948966) q[29], q[27];
cu1(1.5707963267948966) q[16], q[13];
crz(1.5707963267948966) q[10], q[6];
crz(1.5707963267948966) q[18], q[5];
cz q[9], q[11];
crz(1.5707963267948966) q[7], q[1];
cu1(1.5707963267948966) q[12], q[13];
cz q[10], q[20];
cu1(1.5707963267948966) q[14], q[6];
crz(1.5707963267948966) q[1], q[8];
cu1(1.5707963267948966) q[19], q[15];
cu1(1.5707963267948966) q[16], q[27];
cz q[25], q[0];
cu1(1.5707963267948966) q[6], q[0];
crz(1.5707963267948966) q[19], q[25];
cu1(1.5707963267948966) q[19], q[27];
cz q[8], q[15];
cz q[20], q[21];
cu1(1.5707963267948966) q[21], q[16];
crz(1.5707963267948966) q[19], q[5];
crz(1.5707963267948966) q[18], q[1];
crz(1.5707963267948966) q[22], q[28];
cu1(1.5707963267948966) q[17], q[28];
crz(1.5707963267948966) q[8], q[10];
crz(1.5707963267948966) q[18], q[10];
cu1(1.5707963267948966) q[26], q[1];
cz q[11], q[8];
cz q[11], q[17];
cz q[18], q[29];
cz q[3], q[6];
cu1(1.5707963267948966) q[8], q[19];
crz(1.5707963267948966) q[2], q[1];
cu1(1.5707963267948966) q[16], q[25];
crz(1.5707963267948966) q[10], q[13];
crz(1.5707963267948966) q[19], q[3];
crz(1.5707963267948966) q[21], q[4];
cz q[22], q[24];
cz q[27], q[24];
crz(1.5707963267948966) q[15], q[9];
crz(1.5707963267948966) q[4], q[29];
crz(1.5707963267948966) q[16], q[14];
cz q[8], q[0];
crz(1.5707963267948966) q[26], q[23];
cz q[12], q[20];
cu1(1.5707963267948966) q[22], q[18];
cz q[15], q[28];
crz(1.5707963267948966) q[5], q[0];
cz q[15], q[14];
crz(1.5707963267948966) q[19], q[23];
cu1(1.5707963267948966) q[17], q[5];
crz(1.5707963267948966) q[7], q[14];
cu1(1.5707963267948966) q[24], q[29];
cu1(1.5707963267948966) q[5], q[4];
crz(1.5707963267948966) q[29], q[18];
crz(1.5707963267948966) q[8], q[29];
cu1(1.5707963267948966) q[25], q[16];
crz(1.5707963267948966) q[23], q[24];
crz(1.5707963267948966) q[8], q[21];
