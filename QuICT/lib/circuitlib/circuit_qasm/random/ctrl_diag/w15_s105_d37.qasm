OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
crz(1.5707963267948966) q[11], q[9];
cz q[10], q[4];
cu1(1.5707963267948966) q[5], q[13];
crz(1.5707963267948966) q[12], q[4];
cz q[6], q[1];
crz(1.5707963267948966) q[4], q[3];
crz(1.5707963267948966) q[14], q[13];
cz q[5], q[3];
cz q[10], q[14];
crz(1.5707963267948966) q[13], q[8];
cz q[14], q[4];
crz(1.5707963267948966) q[3], q[0];
cu1(1.5707963267948966) q[14], q[13];
cu1(1.5707963267948966) q[0], q[5];
cz q[12], q[0];
cz q[11], q[10];
cz q[8], q[13];
crz(1.5707963267948966) q[6], q[8];
cz q[11], q[3];
cu1(1.5707963267948966) q[11], q[8];
cu1(1.5707963267948966) q[13], q[7];
cu1(1.5707963267948966) q[12], q[1];
cu1(1.5707963267948966) q[6], q[3];
crz(1.5707963267948966) q[7], q[10];
cz q[2], q[8];
cz q[10], q[11];
cu1(1.5707963267948966) q[4], q[2];
crz(1.5707963267948966) q[5], q[1];
crz(1.5707963267948966) q[12], q[5];
cz q[13], q[2];
cz q[12], q[7];
cu1(1.5707963267948966) q[8], q[7];
cu1(1.5707963267948966) q[5], q[10];
cu1(1.5707963267948966) q[1], q[11];
cz q[11], q[14];
crz(1.5707963267948966) q[12], q[9];
cu1(1.5707963267948966) q[0], q[14];
cz q[7], q[11];
cz q[11], q[8];
crz(1.5707963267948966) q[6], q[13];
crz(1.5707963267948966) q[12], q[14];
cu1(1.5707963267948966) q[7], q[11];
cz q[12], q[6];
cz q[7], q[1];
cu1(1.5707963267948966) q[8], q[11];
cu1(1.5707963267948966) q[12], q[3];
cu1(1.5707963267948966) q[6], q[8];
cz q[3], q[10];
cz q[4], q[10];
cu1(1.5707963267948966) q[2], q[1];
cz q[6], q[3];
cu1(1.5707963267948966) q[3], q[0];
cz q[9], q[14];
cz q[14], q[2];
cz q[8], q[14];
cu1(1.5707963267948966) q[5], q[4];
cu1(1.5707963267948966) q[9], q[13];
cu1(1.5707963267948966) q[7], q[6];
crz(1.5707963267948966) q[11], q[8];
cu1(1.5707963267948966) q[10], q[7];
cu1(1.5707963267948966) q[7], q[0];
cz q[2], q[11];
cz q[0], q[14];
cz q[12], q[5];
crz(1.5707963267948966) q[0], q[11];
crz(1.5707963267948966) q[3], q[7];
cu1(1.5707963267948966) q[9], q[7];
cu1(1.5707963267948966) q[3], q[6];
crz(1.5707963267948966) q[12], q[4];
cu1(1.5707963267948966) q[13], q[12];
crz(1.5707963267948966) q[13], q[14];
cu1(1.5707963267948966) q[7], q[8];
cu1(1.5707963267948966) q[12], q[6];
crz(1.5707963267948966) q[5], q[14];
cz q[9], q[14];
cz q[5], q[4];
crz(1.5707963267948966) q[4], q[6];
cu1(1.5707963267948966) q[9], q[10];
cz q[1], q[3];
cu1(1.5707963267948966) q[1], q[2];
cz q[11], q[9];
cz q[1], q[10];
crz(1.5707963267948966) q[13], q[0];
cz q[8], q[0];
crz(1.5707963267948966) q[10], q[9];
cz q[8], q[5];
cu1(1.5707963267948966) q[9], q[4];
cu1(1.5707963267948966) q[10], q[0];
crz(1.5707963267948966) q[14], q[3];
crz(1.5707963267948966) q[8], q[13];
cz q[2], q[6];
crz(1.5707963267948966) q[6], q[12];
crz(1.5707963267948966) q[5], q[4];
cz q[8], q[5];
cz q[8], q[7];
cz q[8], q[12];
crz(1.5707963267948966) q[11], q[13];
crz(1.5707963267948966) q[0], q[1];
cu1(1.5707963267948966) q[13], q[2];
cz q[13], q[8];
crz(1.5707963267948966) q[12], q[8];
cu1(1.5707963267948966) q[9], q[8];
cu1(1.5707963267948966) q[5], q[1];
cz q[10], q[8];
cu1(1.5707963267948966) q[6], q[9];
