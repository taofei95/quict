OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
crz(1.5707963267948966) q[26], q[12];
cz q[5], q[17];
crz(1.5707963267948966) q[18], q[4];
crz(1.5707963267948966) q[17], q[22];
crz(1.5707963267948966) q[21], q[15];
crz(1.5707963267948966) q[29], q[4];
crz(1.5707963267948966) q[8], q[2];
crz(1.5707963267948966) q[29], q[21];
cu1(1.5707963267948966) q[0], q[3];
crz(1.5707963267948966) q[17], q[21];
crz(1.5707963267948966) q[9], q[5];
crz(1.5707963267948966) q[26], q[8];
crz(1.5707963267948966) q[16], q[22];
cz q[12], q[8];
crz(1.5707963267948966) q[1], q[19];
cu1(1.5707963267948966) q[14], q[15];
cz q[3], q[28];
crz(1.5707963267948966) q[16], q[18];
cz q[28], q[6];
crz(1.5707963267948966) q[4], q[13];
crz(1.5707963267948966) q[0], q[25];
crz(1.5707963267948966) q[26], q[5];
cz q[27], q[0];
crz(1.5707963267948966) q[9], q[16];
cz q[9], q[25];
cz q[11], q[20];
crz(1.5707963267948966) q[22], q[8];
cz q[25], q[24];
cu1(1.5707963267948966) q[24], q[10];
cu1(1.5707963267948966) q[8], q[15];
cu1(1.5707963267948966) q[19], q[18];
cz q[28], q[1];
cu1(1.5707963267948966) q[12], q[24];
crz(1.5707963267948966) q[15], q[18];
cu1(1.5707963267948966) q[2], q[22];
cz q[8], q[21];
cu1(1.5707963267948966) q[16], q[9];
crz(1.5707963267948966) q[5], q[22];
cu1(1.5707963267948966) q[2], q[20];
cu1(1.5707963267948966) q[24], q[0];
crz(1.5707963267948966) q[11], q[16];
crz(1.5707963267948966) q[1], q[25];
crz(1.5707963267948966) q[19], q[15];
crz(1.5707963267948966) q[11], q[22];
cu1(1.5707963267948966) q[5], q[20];
cz q[27], q[19];
cu1(1.5707963267948966) q[9], q[23];
crz(1.5707963267948966) q[16], q[6];
cu1(1.5707963267948966) q[3], q[16];
crz(1.5707963267948966) q[4], q[18];
cz q[4], q[26];
cu1(1.5707963267948966) q[9], q[24];
cz q[16], q[13];
crz(1.5707963267948966) q[19], q[21];
cu1(1.5707963267948966) q[27], q[3];
crz(1.5707963267948966) q[24], q[29];
cu1(1.5707963267948966) q[29], q[15];
cu1(1.5707963267948966) q[22], q[9];
cu1(1.5707963267948966) q[17], q[8];
cu1(1.5707963267948966) q[21], q[25];
cu1(1.5707963267948966) q[11], q[28];
crz(1.5707963267948966) q[19], q[15];
crz(1.5707963267948966) q[7], q[1];
crz(1.5707963267948966) q[4], q[8];
cz q[3], q[8];
crz(1.5707963267948966) q[3], q[28];
cz q[22], q[18];
cz q[13], q[28];
cu1(1.5707963267948966) q[15], q[23];
crz(1.5707963267948966) q[24], q[6];
crz(1.5707963267948966) q[22], q[6];
cu1(1.5707963267948966) q[8], q[16];
cz q[10], q[0];
cz q[17], q[10];
crz(1.5707963267948966) q[10], q[17];
cu1(1.5707963267948966) q[16], q[21];
crz(1.5707963267948966) q[25], q[15];
cz q[19], q[9];
cz q[17], q[15];
cu1(1.5707963267948966) q[10], q[21];
cz q[17], q[6];
cu1(1.5707963267948966) q[0], q[17];
cu1(1.5707963267948966) q[26], q[1];
cu1(1.5707963267948966) q[23], q[29];
crz(1.5707963267948966) q[17], q[8];
cu1(1.5707963267948966) q[9], q[19];
cz q[13], q[26];
crz(1.5707963267948966) q[17], q[22];
crz(1.5707963267948966) q[23], q[26];
cz q[9], q[22];
cu1(1.5707963267948966) q[23], q[20];
cu1(1.5707963267948966) q[21], q[26];
cu1(1.5707963267948966) q[0], q[8];
crz(1.5707963267948966) q[2], q[19];
cu1(1.5707963267948966) q[8], q[21];
crz(1.5707963267948966) q[14], q[3];
crz(1.5707963267948966) q[23], q[18];
cz q[22], q[11];
cz q[21], q[24];
crz(1.5707963267948966) q[19], q[7];
cu1(1.5707963267948966) q[0], q[6];
cu1(1.5707963267948966) q[14], q[24];
crz(1.5707963267948966) q[28], q[17];
cu1(1.5707963267948966) q[23], q[9];
cu1(1.5707963267948966) q[5], q[1];
cu1(1.5707963267948966) q[3], q[12];
crz(1.5707963267948966) q[7], q[2];
cz q[2], q[29];
cz q[25], q[4];
cz q[2], q[18];
cu1(1.5707963267948966) q[10], q[5];
cz q[7], q[2];
crz(1.5707963267948966) q[24], q[8];
cz q[17], q[10];
crz(1.5707963267948966) q[27], q[6];
cu1(1.5707963267948966) q[2], q[17];
cz q[17], q[2];
cz q[0], q[17];
crz(1.5707963267948966) q[27], q[12];
cz q[7], q[2];
cz q[11], q[7];
cz q[24], q[17];
cu1(1.5707963267948966) q[4], q[1];
cz q[13], q[29];
cz q[18], q[26];
crz(1.5707963267948966) q[13], q[27];
crz(1.5707963267948966) q[19], q[17];
cz q[13], q[3];
cu1(1.5707963267948966) q[7], q[17];
crz(1.5707963267948966) q[5], q[23];
cz q[13], q[16];
cu1(1.5707963267948966) q[17], q[23];
crz(1.5707963267948966) q[3], q[5];
crz(1.5707963267948966) q[20], q[21];
cu1(1.5707963267948966) q[0], q[4];
cu1(1.5707963267948966) q[0], q[1];
cu1(1.5707963267948966) q[15], q[29];
cz q[11], q[16];
crz(1.5707963267948966) q[29], q[6];
cu1(1.5707963267948966) q[9], q[16];
cz q[1], q[23];
cz q[24], q[21];
cz q[15], q[21];
cz q[29], q[27];
cu1(1.5707963267948966) q[17], q[12];
crz(1.5707963267948966) q[29], q[13];
cz q[20], q[29];
cz q[6], q[18];
cz q[1], q[3];
crz(1.5707963267948966) q[14], q[16];
