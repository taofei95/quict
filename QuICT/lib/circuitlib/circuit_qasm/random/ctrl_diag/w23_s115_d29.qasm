OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
cu1(1.5707963267948966) q[10], q[16];
cu1(1.5707963267948966) q[5], q[13];
cz q[12], q[18];
cz q[16], q[0];
crz(1.5707963267948966) q[15], q[9];
crz(1.5707963267948966) q[15], q[17];
crz(1.5707963267948966) q[17], q[18];
cu1(1.5707963267948966) q[7], q[8];
cz q[19], q[14];
cz q[19], q[7];
cu1(1.5707963267948966) q[0], q[5];
cu1(1.5707963267948966) q[3], q[15];
crz(1.5707963267948966) q[19], q[3];
cz q[15], q[13];
cz q[13], q[15];
cu1(1.5707963267948966) q[2], q[17];
cu1(1.5707963267948966) q[1], q[3];
cu1(1.5707963267948966) q[18], q[17];
cu1(1.5707963267948966) q[18], q[2];
cu1(1.5707963267948966) q[11], q[2];
cz q[17], q[0];
crz(1.5707963267948966) q[14], q[9];
cz q[7], q[4];
crz(1.5707963267948966) q[8], q[20];
cu1(1.5707963267948966) q[12], q[22];
cz q[2], q[15];
cz q[12], q[22];
crz(1.5707963267948966) q[14], q[9];
cz q[2], q[21];
crz(1.5707963267948966) q[3], q[18];
crz(1.5707963267948966) q[20], q[11];
cz q[0], q[18];
crz(1.5707963267948966) q[11], q[7];
cz q[7], q[4];
cu1(1.5707963267948966) q[3], q[7];
cu1(1.5707963267948966) q[19], q[0];
cu1(1.5707963267948966) q[19], q[4];
crz(1.5707963267948966) q[1], q[11];
cz q[7], q[18];
cu1(1.5707963267948966) q[10], q[20];
crz(1.5707963267948966) q[8], q[16];
cu1(1.5707963267948966) q[10], q[4];
cz q[10], q[4];
crz(1.5707963267948966) q[20], q[12];
cu1(1.5707963267948966) q[11], q[13];
cu1(1.5707963267948966) q[9], q[7];
crz(1.5707963267948966) q[5], q[1];
crz(1.5707963267948966) q[10], q[7];
crz(1.5707963267948966) q[4], q[0];
crz(1.5707963267948966) q[5], q[8];
crz(1.5707963267948966) q[7], q[20];
cu1(1.5707963267948966) q[18], q[14];
cu1(1.5707963267948966) q[20], q[0];
cz q[19], q[22];
crz(1.5707963267948966) q[6], q[7];
cz q[14], q[22];
crz(1.5707963267948966) q[22], q[13];
cu1(1.5707963267948966) q[14], q[22];
crz(1.5707963267948966) q[16], q[2];
cu1(1.5707963267948966) q[17], q[14];
cu1(1.5707963267948966) q[9], q[6];
cu1(1.5707963267948966) q[4], q[10];
cz q[16], q[10];
cu1(1.5707963267948966) q[8], q[22];
crz(1.5707963267948966) q[21], q[17];
cu1(1.5707963267948966) q[9], q[14];
cz q[7], q[10];
crz(1.5707963267948966) q[21], q[10];
crz(1.5707963267948966) q[19], q[7];
cz q[3], q[13];
cz q[9], q[3];
cz q[14], q[16];
crz(1.5707963267948966) q[14], q[0];
crz(1.5707963267948966) q[21], q[1];
crz(1.5707963267948966) q[5], q[12];
crz(1.5707963267948966) q[17], q[20];
crz(1.5707963267948966) q[21], q[17];
crz(1.5707963267948966) q[14], q[18];
crz(1.5707963267948966) q[0], q[15];
cz q[7], q[10];
cu1(1.5707963267948966) q[14], q[19];
crz(1.5707963267948966) q[21], q[14];
cz q[0], q[12];
cu1(1.5707963267948966) q[9], q[1];
cu1(1.5707963267948966) q[9], q[17];
cu1(1.5707963267948966) q[12], q[1];
cu1(1.5707963267948966) q[9], q[6];
cz q[1], q[15];
cu1(1.5707963267948966) q[19], q[4];
crz(1.5707963267948966) q[21], q[17];
crz(1.5707963267948966) q[22], q[0];
cu1(1.5707963267948966) q[5], q[1];
cu1(1.5707963267948966) q[20], q[10];
cz q[14], q[18];
crz(1.5707963267948966) q[9], q[0];
cz q[10], q[20];
crz(1.5707963267948966) q[7], q[12];
crz(1.5707963267948966) q[9], q[22];
cu1(1.5707963267948966) q[9], q[11];
cz q[8], q[12];
cz q[3], q[10];
cu1(1.5707963267948966) q[2], q[21];
crz(1.5707963267948966) q[7], q[5];
crz(1.5707963267948966) q[15], q[3];
cz q[3], q[5];
crz(1.5707963267948966) q[20], q[7];
cz q[19], q[13];
cz q[2], q[5];
crz(1.5707963267948966) q[18], q[2];
cu1(1.5707963267948966) q[8], q[7];
cu1(1.5707963267948966) q[0], q[21];
cu1(1.5707963267948966) q[11], q[13];
crz(1.5707963267948966) q[19], q[8];
cz q[6], q[10];
cu1(1.5707963267948966) q[10], q[16];
