OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
cz q[3], q[14];
cz q[17], q[3];
cu1(1.5707963267948966) q[11], q[5];
cz q[22], q[9];
crz(1.5707963267948966) q[22], q[21];
cu1(1.5707963267948966) q[10], q[20];
crz(1.5707963267948966) q[10], q[11];
cu1(1.5707963267948966) q[0], q[15];
crz(1.5707963267948966) q[2], q[9];
cz q[9], q[19];
cz q[4], q[23];
crz(1.5707963267948966) q[8], q[12];
cz q[3], q[14];
cz q[5], q[6];
cz q[21], q[8];
crz(1.5707963267948966) q[17], q[3];
crz(1.5707963267948966) q[2], q[20];
cz q[16], q[12];
crz(1.5707963267948966) q[5], q[14];
cu1(1.5707963267948966) q[2], q[12];
cz q[14], q[10];
crz(1.5707963267948966) q[21], q[19];
cz q[11], q[15];
cz q[1], q[11];
crz(1.5707963267948966) q[6], q[11];
crz(1.5707963267948966) q[20], q[7];
cu1(1.5707963267948966) q[4], q[16];
cz q[18], q[1];
crz(1.5707963267948966) q[13], q[10];
cz q[5], q[2];
cz q[10], q[23];
crz(1.5707963267948966) q[15], q[8];
crz(1.5707963267948966) q[9], q[1];
cz q[8], q[17];
cz q[23], q[18];
crz(1.5707963267948966) q[5], q[22];
crz(1.5707963267948966) q[5], q[20];
cu1(1.5707963267948966) q[10], q[18];
crz(1.5707963267948966) q[11], q[7];
cz q[22], q[10];
cz q[20], q[10];
cu1(1.5707963267948966) q[12], q[0];
crz(1.5707963267948966) q[2], q[21];
crz(1.5707963267948966) q[13], q[3];
cz q[11], q[2];
cz q[12], q[2];
cz q[17], q[10];
cz q[5], q[9];
crz(1.5707963267948966) q[5], q[7];
cz q[11], q[8];
cz q[6], q[5];
cu1(1.5707963267948966) q[16], q[9];
crz(1.5707963267948966) q[21], q[11];
crz(1.5707963267948966) q[16], q[13];
crz(1.5707963267948966) q[18], q[11];
cz q[6], q[20];
cu1(1.5707963267948966) q[21], q[2];
cz q[11], q[0];
cz q[9], q[1];
cu1(1.5707963267948966) q[3], q[21];
cz q[2], q[17];
crz(1.5707963267948966) q[14], q[13];
cu1(1.5707963267948966) q[23], q[1];
cu1(1.5707963267948966) q[9], q[17];
cz q[3], q[1];
crz(1.5707963267948966) q[22], q[8];
cu1(1.5707963267948966) q[2], q[11];
cz q[10], q[3];
crz(1.5707963267948966) q[17], q[0];
cz q[6], q[0];
cz q[3], q[19];
cz q[18], q[0];
crz(1.5707963267948966) q[4], q[16];
crz(1.5707963267948966) q[1], q[23];
crz(1.5707963267948966) q[16], q[14];
cu1(1.5707963267948966) q[4], q[1];
cz q[23], q[13];
crz(1.5707963267948966) q[13], q[11];
crz(1.5707963267948966) q[3], q[14];
crz(1.5707963267948966) q[10], q[16];
cu1(1.5707963267948966) q[8], q[19];
cz q[4], q[19];
cz q[20], q[19];
cu1(1.5707963267948966) q[10], q[3];
cu1(1.5707963267948966) q[21], q[6];
cz q[7], q[12];
crz(1.5707963267948966) q[8], q[21];
cu1(1.5707963267948966) q[17], q[11];
crz(1.5707963267948966) q[12], q[13];
cu1(1.5707963267948966) q[3], q[8];
cu1(1.5707963267948966) q[17], q[13];
cu1(1.5707963267948966) q[14], q[20];
crz(1.5707963267948966) q[8], q[2];
crz(1.5707963267948966) q[15], q[17];
crz(1.5707963267948966) q[14], q[13];
cz q[21], q[10];
cz q[3], q[0];
cu1(1.5707963267948966) q[20], q[22];
cu1(1.5707963267948966) q[13], q[1];
crz(1.5707963267948966) q[20], q[1];
cu1(1.5707963267948966) q[22], q[12];
cz q[19], q[20];
cz q[22], q[17];
cu1(1.5707963267948966) q[15], q[3];
crz(1.5707963267948966) q[6], q[13];
cu1(1.5707963267948966) q[6], q[23];
cz q[18], q[14];
cz q[12], q[9];
crz(1.5707963267948966) q[10], q[12];
crz(1.5707963267948966) q[8], q[12];
cz q[16], q[19];
cz q[20], q[6];
cz q[13], q[2];
crz(1.5707963267948966) q[10], q[14];
cz q[2], q[6];
cu1(1.5707963267948966) q[16], q[10];
cu1(1.5707963267948966) q[0], q[19];
cz q[14], q[3];
cu1(1.5707963267948966) q[16], q[8];
cz q[11], q[19];
crz(1.5707963267948966) q[3], q[8];
cz q[19], q[14];
crz(1.5707963267948966) q[1], q[13];
cz q[9], q[3];
crz(1.5707963267948966) q[21], q[18];
crz(1.5707963267948966) q[11], q[17];
cu1(1.5707963267948966) q[3], q[6];
cu1(1.5707963267948966) q[10], q[17];
cu1(1.5707963267948966) q[14], q[16];
cu1(1.5707963267948966) q[11], q[21];
crz(1.5707963267948966) q[13], q[18];
crz(1.5707963267948966) q[12], q[5];
cu1(1.5707963267948966) q[4], q[1];
cz q[23], q[5];
cu1(1.5707963267948966) q[18], q[21];
crz(1.5707963267948966) q[8], q[20];
cu1(1.5707963267948966) q[22], q[0];
cu1(1.5707963267948966) q[3], q[6];
cu1(1.5707963267948966) q[0], q[10];
crz(1.5707963267948966) q[4], q[23];
crz(1.5707963267948966) q[5], q[14];
cz q[6], q[7];
cu1(1.5707963267948966) q[15], q[7];
cu1(1.5707963267948966) q[16], q[22];
crz(1.5707963267948966) q[23], q[2];
crz(1.5707963267948966) q[22], q[7];
cz q[9], q[13];
cz q[0], q[16];
cu1(1.5707963267948966) q[23], q[17];
cu1(1.5707963267948966) q[15], q[16];
cu1(1.5707963267948966) q[11], q[0];
cz q[11], q[22];
crz(1.5707963267948966) q[0], q[7];
cu1(1.5707963267948966) q[11], q[15];
crz(1.5707963267948966) q[8], q[11];
crz(1.5707963267948966) q[4], q[23];
crz(1.5707963267948966) q[6], q[11];
cu1(1.5707963267948966) q[2], q[17];
cz q[1], q[20];
crz(1.5707963267948966) q[6], q[14];
cu1(1.5707963267948966) q[13], q[17];
cz q[12], q[19];
cz q[18], q[4];
cz q[19], q[22];
cz q[7], q[0];
cz q[6], q[7];
cu1(1.5707963267948966) q[12], q[17];
cu1(1.5707963267948966) q[0], q[6];
cz q[14], q[10];
crz(1.5707963267948966) q[20], q[21];
crz(1.5707963267948966) q[6], q[12];
crz(1.5707963267948966) q[8], q[10];
cz q[11], q[15];
cu1(1.5707963267948966) q[8], q[6];
cu1(1.5707963267948966) q[18], q[13];
cz q[9], q[7];
cu1(1.5707963267948966) q[12], q[0];
crz(1.5707963267948966) q[1], q[6];
crz(1.5707963267948966) q[22], q[3];
cz q[19], q[1];
crz(1.5707963267948966) q[7], q[10];
cu1(1.5707963267948966) q[21], q[15];
crz(1.5707963267948966) q[20], q[17];
crz(1.5707963267948966) q[2], q[17];
cz q[0], q[2];
cz q[6], q[21];
cz q[2], q[14];
cu1(1.5707963267948966) q[3], q[22];
crz(1.5707963267948966) q[1], q[10];
cu1(1.5707963267948966) q[5], q[13];
crz(1.5707963267948966) q[18], q[6];
crz(1.5707963267948966) q[9], q[0];
