OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
cz q[13], q[0];
crz(1.5707963267948966) q[13], q[1];
cu1(1.5707963267948966) q[0], q[4];
cz q[19], q[14];
cu1(1.5707963267948966) q[14], q[9];
cu1(1.5707963267948966) q[7], q[0];
crz(1.5707963267948966) q[20], q[16];
cz q[1], q[6];
cu1(1.5707963267948966) q[21], q[15];
cz q[21], q[12];
cz q[6], q[7];
crz(1.5707963267948966) q[21], q[11];
cz q[8], q[17];
cz q[8], q[6];
crz(1.5707963267948966) q[6], q[11];
cz q[13], q[17];
cz q[21], q[5];
crz(1.5707963267948966) q[12], q[20];
cu1(1.5707963267948966) q[9], q[2];
cu1(1.5707963267948966) q[3], q[16];
cz q[5], q[17];
crz(1.5707963267948966) q[9], q[18];
cu1(1.5707963267948966) q[17], q[11];
cz q[9], q[16];
cu1(1.5707963267948966) q[12], q[22];
cu1(1.5707963267948966) q[16], q[7];
cz q[15], q[11];
crz(1.5707963267948966) q[13], q[11];
crz(1.5707963267948966) q[8], q[13];
cz q[20], q[4];
cz q[3], q[0];
cz q[3], q[13];
cz q[19], q[14];
crz(1.5707963267948966) q[0], q[17];
cz q[14], q[20];
crz(1.5707963267948966) q[21], q[17];
crz(1.5707963267948966) q[16], q[7];
cu1(1.5707963267948966) q[4], q[14];
cz q[15], q[8];
cu1(1.5707963267948966) q[14], q[13];
crz(1.5707963267948966) q[7], q[3];
cz q[7], q[1];
cu1(1.5707963267948966) q[15], q[8];
cz q[11], q[8];
cz q[7], q[1];
cz q[2], q[0];
cu1(1.5707963267948966) q[22], q[11];
cz q[19], q[3];
cu1(1.5707963267948966) q[10], q[1];
cz q[18], q[19];
cz q[22], q[20];
crz(1.5707963267948966) q[19], q[1];
cz q[6], q[7];
cz q[3], q[2];
cz q[0], q[14];
cu1(1.5707963267948966) q[13], q[9];
cz q[16], q[12];
cu1(1.5707963267948966) q[1], q[8];
cu1(1.5707963267948966) q[11], q[15];
cu1(1.5707963267948966) q[15], q[5];
crz(1.5707963267948966) q[3], q[2];
cu1(1.5707963267948966) q[15], q[19];
cu1(1.5707963267948966) q[15], q[1];
cz q[22], q[13];
cu1(1.5707963267948966) q[1], q[14];
cz q[14], q[22];
crz(1.5707963267948966) q[11], q[14];
crz(1.5707963267948966) q[18], q[20];
crz(1.5707963267948966) q[2], q[10];
cz q[12], q[8];
cz q[2], q[6];
cz q[2], q[3];
crz(1.5707963267948966) q[16], q[1];
cz q[11], q[4];
crz(1.5707963267948966) q[2], q[16];
crz(1.5707963267948966) q[15], q[7];
crz(1.5707963267948966) q[2], q[20];
crz(1.5707963267948966) q[4], q[1];
crz(1.5707963267948966) q[2], q[8];
cu1(1.5707963267948966) q[18], q[7];
cz q[20], q[19];
cz q[14], q[11];
cz q[16], q[1];
crz(1.5707963267948966) q[0], q[16];
cu1(1.5707963267948966) q[20], q[12];
cz q[14], q[15];
crz(1.5707963267948966) q[4], q[6];
crz(1.5707963267948966) q[5], q[15];
cz q[12], q[15];
cu1(1.5707963267948966) q[22], q[18];
cz q[1], q[3];
cu1(1.5707963267948966) q[16], q[8];
crz(1.5707963267948966) q[20], q[8];
crz(1.5707963267948966) q[12], q[10];
crz(1.5707963267948966) q[3], q[11];
cu1(1.5707963267948966) q[15], q[7];
cz q[0], q[16];
cz q[13], q[7];
cz q[9], q[12];
cu1(1.5707963267948966) q[4], q[20];
cu1(1.5707963267948966) q[10], q[13];
crz(1.5707963267948966) q[10], q[12];
cu1(1.5707963267948966) q[16], q[3];
cu1(1.5707963267948966) q[9], q[20];
crz(1.5707963267948966) q[10], q[9];
cz q[4], q[18];
cu1(1.5707963267948966) q[6], q[16];
cz q[17], q[20];
cu1(1.5707963267948966) q[5], q[6];
crz(1.5707963267948966) q[4], q[12];
cz q[14], q[11];
crz(1.5707963267948966) q[6], q[19];
cu1(1.5707963267948966) q[18], q[21];
cu1(1.5707963267948966) q[20], q[16];
cu1(1.5707963267948966) q[22], q[17];
cu1(1.5707963267948966) q[17], q[18];
cz q[18], q[11];
cu1(1.5707963267948966) q[5], q[8];
crz(1.5707963267948966) q[20], q[2];
cu1(1.5707963267948966) q[20], q[9];
cz q[13], q[22];
crz(1.5707963267948966) q[12], q[3];
cz q[15], q[4];
crz(1.5707963267948966) q[15], q[12];
cz q[21], q[9];
cu1(1.5707963267948966) q[15], q[6];
cu1(1.5707963267948966) q[13], q[4];
crz(1.5707963267948966) q[17], q[5];
cz q[2], q[6];
crz(1.5707963267948966) q[15], q[8];
cz q[15], q[10];
crz(1.5707963267948966) q[12], q[15];
crz(1.5707963267948966) q[21], q[9];
crz(1.5707963267948966) q[22], q[5];
crz(1.5707963267948966) q[13], q[20];
crz(1.5707963267948966) q[16], q[8];
crz(1.5707963267948966) q[13], q[18];
cu1(1.5707963267948966) q[14], q[11];
cu1(1.5707963267948966) q[4], q[17];
crz(1.5707963267948966) q[12], q[21];
cu1(1.5707963267948966) q[8], q[21];
cu1(1.5707963267948966) q[8], q[18];
crz(1.5707963267948966) q[6], q[20];
cu1(1.5707963267948966) q[16], q[14];
crz(1.5707963267948966) q[20], q[3];
crz(1.5707963267948966) q[20], q[15];
cu1(1.5707963267948966) q[7], q[9];
cz q[20], q[17];
cu1(1.5707963267948966) q[13], q[19];
crz(1.5707963267948966) q[8], q[9];
cu1(1.5707963267948966) q[12], q[8];
cu1(1.5707963267948966) q[16], q[10];
cu1(1.5707963267948966) q[19], q[22];
cz q[1], q[16];
crz(1.5707963267948966) q[17], q[22];
crz(1.5707963267948966) q[10], q[4];
cu1(1.5707963267948966) q[5], q[21];
cu1(1.5707963267948966) q[1], q[8];
crz(1.5707963267948966) q[14], q[11];
cu1(1.5707963267948966) q[3], q[15];
cz q[9], q[8];
cu1(1.5707963267948966) q[0], q[20];
cu1(1.5707963267948966) q[12], q[4];
crz(1.5707963267948966) q[2], q[15];
crz(1.5707963267948966) q[16], q[13];
cu1(1.5707963267948966) q[1], q[14];
cu1(1.5707963267948966) q[0], q[1];
cz q[19], q[20];
cu1(1.5707963267948966) q[18], q[6];
crz(1.5707963267948966) q[2], q[17];
cz q[2], q[18];
cz q[13], q[2];
crz(1.5707963267948966) q[13], q[11];
cz q[7], q[1];
crz(1.5707963267948966) q[17], q[15];
cz q[21], q[1];
crz(1.5707963267948966) q[20], q[11];
cz q[14], q[18];
cz q[8], q[10];
crz(1.5707963267948966) q[7], q[12];
crz(1.5707963267948966) q[14], q[2];
cu1(1.5707963267948966) q[20], q[15];
cu1(1.5707963267948966) q[22], q[12];
cz q[0], q[20];
