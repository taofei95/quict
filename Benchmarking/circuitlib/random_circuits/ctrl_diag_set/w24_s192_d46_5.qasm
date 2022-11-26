OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
cu1(1.5707963267948966) q[17], q[22];
cz q[5], q[12];
cz q[21], q[23];
crz(1.5707963267948966) q[18], q[20];
cz q[8], q[13];
crz(1.5707963267948966) q[18], q[11];
cu1(1.5707963267948966) q[3], q[14];
cz q[3], q[13];
cu1(1.5707963267948966) q[14], q[16];
cu1(1.5707963267948966) q[2], q[4];
cu1(1.5707963267948966) q[20], q[0];
crz(1.5707963267948966) q[1], q[10];
crz(1.5707963267948966) q[23], q[19];
crz(1.5707963267948966) q[10], q[16];
cu1(1.5707963267948966) q[22], q[8];
crz(1.5707963267948966) q[10], q[17];
cz q[18], q[3];
crz(1.5707963267948966) q[14], q[21];
crz(1.5707963267948966) q[4], q[23];
crz(1.5707963267948966) q[6], q[0];
crz(1.5707963267948966) q[6], q[4];
cz q[13], q[15];
cu1(1.5707963267948966) q[7], q[5];
cu1(1.5707963267948966) q[10], q[4];
crz(1.5707963267948966) q[22], q[1];
crz(1.5707963267948966) q[3], q[13];
crz(1.5707963267948966) q[11], q[9];
cu1(1.5707963267948966) q[19], q[22];
cz q[1], q[2];
cz q[17], q[12];
crz(1.5707963267948966) q[14], q[15];
cu1(1.5707963267948966) q[8], q[13];
cz q[3], q[15];
cu1(1.5707963267948966) q[20], q[6];
crz(1.5707963267948966) q[14], q[15];
cz q[22], q[12];
crz(1.5707963267948966) q[15], q[20];
crz(1.5707963267948966) q[15], q[22];
crz(1.5707963267948966) q[2], q[7];
cu1(1.5707963267948966) q[2], q[14];
cz q[8], q[0];
cu1(1.5707963267948966) q[10], q[21];
cz q[8], q[1];
crz(1.5707963267948966) q[9], q[22];
cu1(1.5707963267948966) q[2], q[3];
cz q[0], q[7];
crz(1.5707963267948966) q[14], q[13];
cu1(1.5707963267948966) q[21], q[5];
crz(1.5707963267948966) q[2], q[0];
cz q[0], q[18];
cu1(1.5707963267948966) q[6], q[15];
cu1(1.5707963267948966) q[8], q[5];
cz q[20], q[15];
crz(1.5707963267948966) q[13], q[23];
cz q[14], q[22];
cz q[8], q[4];
crz(1.5707963267948966) q[4], q[12];
cu1(1.5707963267948966) q[2], q[5];
crz(1.5707963267948966) q[17], q[6];
cz q[11], q[14];
cu1(1.5707963267948966) q[10], q[22];
cz q[20], q[4];
cu1(1.5707963267948966) q[4], q[3];
cz q[11], q[1];
cu1(1.5707963267948966) q[20], q[15];
cu1(1.5707963267948966) q[13], q[17];
cz q[10], q[21];
cz q[11], q[13];
crz(1.5707963267948966) q[7], q[6];
cu1(1.5707963267948966) q[6], q[13];
crz(1.5707963267948966) q[20], q[1];
cz q[5], q[21];
crz(1.5707963267948966) q[11], q[6];
cz q[7], q[5];
crz(1.5707963267948966) q[11], q[19];
cu1(1.5707963267948966) q[18], q[1];
cu1(1.5707963267948966) q[9], q[5];
crz(1.5707963267948966) q[7], q[21];
cz q[15], q[7];
cz q[6], q[11];
crz(1.5707963267948966) q[22], q[6];
crz(1.5707963267948966) q[9], q[16];
cz q[11], q[10];
cz q[23], q[2];
crz(1.5707963267948966) q[20], q[13];
cz q[21], q[12];
cz q[2], q[19];
cz q[8], q[16];
crz(1.5707963267948966) q[20], q[8];
cz q[23], q[19];
crz(1.5707963267948966) q[2], q[22];
cu1(1.5707963267948966) q[7], q[1];
cu1(1.5707963267948966) q[19], q[15];
crz(1.5707963267948966) q[15], q[4];
cz q[11], q[3];
cu1(1.5707963267948966) q[3], q[14];
crz(1.5707963267948966) q[8], q[15];
cz q[0], q[3];
cz q[2], q[10];
cz q[13], q[9];
cz q[22], q[14];
cz q[6], q[18];
cz q[15], q[14];
cu1(1.5707963267948966) q[22], q[15];
cz q[18], q[19];
cu1(1.5707963267948966) q[11], q[21];
cu1(1.5707963267948966) q[21], q[3];
cz q[18], q[17];
cu1(1.5707963267948966) q[1], q[5];
crz(1.5707963267948966) q[5], q[22];
crz(1.5707963267948966) q[11], q[0];
crz(1.5707963267948966) q[17], q[3];
cz q[2], q[21];
cz q[3], q[1];
crz(1.5707963267948966) q[20], q[15];
crz(1.5707963267948966) q[11], q[10];
cz q[17], q[14];
cu1(1.5707963267948966) q[4], q[21];
cz q[7], q[17];
cz q[8], q[17];
cu1(1.5707963267948966) q[1], q[8];
crz(1.5707963267948966) q[22], q[8];
crz(1.5707963267948966) q[5], q[0];
crz(1.5707963267948966) q[14], q[17];
crz(1.5707963267948966) q[17], q[13];
cz q[2], q[3];
crz(1.5707963267948966) q[13], q[20];
cz q[22], q[19];
cz q[1], q[15];
crz(1.5707963267948966) q[1], q[10];
crz(1.5707963267948966) q[22], q[20];
cz q[4], q[11];
crz(1.5707963267948966) q[1], q[5];
cz q[10], q[21];
cu1(1.5707963267948966) q[3], q[8];
cz q[10], q[6];
cu1(1.5707963267948966) q[12], q[11];
crz(1.5707963267948966) q[21], q[4];
cu1(1.5707963267948966) q[1], q[13];
cu1(1.5707963267948966) q[4], q[15];
cz q[4], q[22];
cu1(1.5707963267948966) q[3], q[5];
cu1(1.5707963267948966) q[10], q[5];
cu1(1.5707963267948966) q[8], q[13];
crz(1.5707963267948966) q[3], q[14];
cu1(1.5707963267948966) q[4], q[11];
cu1(1.5707963267948966) q[4], q[15];
crz(1.5707963267948966) q[0], q[13];
cz q[9], q[18];
cu1(1.5707963267948966) q[2], q[20];
cu1(1.5707963267948966) q[11], q[10];
crz(1.5707963267948966) q[21], q[2];
cz q[11], q[1];
cu1(1.5707963267948966) q[2], q[22];
cu1(1.5707963267948966) q[0], q[4];
cu1(1.5707963267948966) q[0], q[19];
cz q[6], q[13];
crz(1.5707963267948966) q[4], q[17];
cz q[5], q[14];
cz q[0], q[22];
cu1(1.5707963267948966) q[13], q[20];
cu1(1.5707963267948966) q[13], q[2];
crz(1.5707963267948966) q[16], q[21];
crz(1.5707963267948966) q[20], q[17];
cz q[13], q[5];
cz q[6], q[20];
cu1(1.5707963267948966) q[23], q[19];
crz(1.5707963267948966) q[17], q[7];
cu1(1.5707963267948966) q[14], q[17];
cz q[12], q[20];
cz q[0], q[18];
cz q[0], q[18];
crz(1.5707963267948966) q[14], q[17];
cz q[19], q[5];
cz q[23], q[5];
cu1(1.5707963267948966) q[19], q[18];
cz q[21], q[16];
cz q[17], q[18];
cz q[8], q[0];
crz(1.5707963267948966) q[4], q[21];
cz q[11], q[21];
cz q[21], q[18];
cz q[11], q[13];
cu1(1.5707963267948966) q[4], q[14];
crz(1.5707963267948966) q[2], q[14];
crz(1.5707963267948966) q[4], q[6];
crz(1.5707963267948966) q[22], q[4];
crz(1.5707963267948966) q[23], q[10];
cu1(1.5707963267948966) q[12], q[5];
crz(1.5707963267948966) q[22], q[8];
crz(1.5707963267948966) q[8], q[2];
crz(1.5707963267948966) q[20], q[5];