OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cu1(1.5707963267948966) q[1], q[6];
cz q[7], q[6];
cz q[0], q[5];
cu1(1.5707963267948966) q[1], q[5];
cu1(1.5707963267948966) q[7], q[5];
cz q[3], q[1];
cz q[0], q[4];
cz q[3], q[6];
cu1(1.5707963267948966) q[0], q[8];
cz q[5], q[4];
cz q[8], q[4];
cu1(1.5707963267948966) q[0], q[7];
cu1(1.5707963267948966) q[6], q[5];
crz(1.5707963267948966) q[4], q[3];
cz q[6], q[5];
crz(1.5707963267948966) q[8], q[3];
crz(1.5707963267948966) q[2], q[0];
cu1(1.5707963267948966) q[5], q[0];
cu1(1.5707963267948966) q[7], q[4];
cz q[6], q[8];
cz q[5], q[7];
cu1(1.5707963267948966) q[1], q[7];
cu1(1.5707963267948966) q[0], q[5];
cz q[6], q[0];
crz(1.5707963267948966) q[2], q[4];
crz(1.5707963267948966) q[0], q[4];
crz(1.5707963267948966) q[0], q[8];
cu1(1.5707963267948966) q[7], q[8];
cu1(1.5707963267948966) q[4], q[1];
cz q[8], q[2];
cz q[4], q[1];
cz q[5], q[6];
crz(1.5707963267948966) q[8], q[4];
cu1(1.5707963267948966) q[3], q[5];
crz(1.5707963267948966) q[0], q[8];
cz q[5], q[7];
crz(1.5707963267948966) q[3], q[2];
crz(1.5707963267948966) q[0], q[5];
cu1(1.5707963267948966) q[5], q[3];
cz q[7], q[4];
cu1(1.5707963267948966) q[5], q[0];
crz(1.5707963267948966) q[2], q[6];
cz q[5], q[3];
crz(1.5707963267948966) q[5], q[0];
cz q[2], q[5];
cz q[7], q[0];
cz q[2], q[3];
crz(1.5707963267948966) q[5], q[7];
cu1(1.5707963267948966) q[2], q[5];
cu1(1.5707963267948966) q[0], q[5];
crz(1.5707963267948966) q[0], q[1];
cz q[6], q[2];
cu1(1.5707963267948966) q[4], q[2];
cz q[4], q[3];
cz q[2], q[3];
cz q[4], q[0];
cu1(1.5707963267948966) q[1], q[5];
cz q[7], q[6];
cu1(1.5707963267948966) q[3], q[4];
cz q[5], q[3];
cz q[7], q[0];
crz(1.5707963267948966) q[5], q[1];
cu1(1.5707963267948966) q[6], q[5];
crz(1.5707963267948966) q[7], q[8];
crz(1.5707963267948966) q[2], q[7];
cu1(1.5707963267948966) q[4], q[3];
cz q[2], q[6];
crz(1.5707963267948966) q[1], q[2];
cu1(1.5707963267948966) q[3], q[7];
cz q[8], q[3];
cu1(1.5707963267948966) q[3], q[2];
crz(1.5707963267948966) q[1], q[3];
cz q[8], q[4];
cu1(1.5707963267948966) q[2], q[5];
cu1(1.5707963267948966) q[7], q[8];
crz(1.5707963267948966) q[7], q[1];
cz q[0], q[8];
cz q[0], q[5];
cz q[1], q[3];
crz(1.5707963267948966) q[0], q[5];
cz q[4], q[2];
cz q[5], q[0];
crz(1.5707963267948966) q[3], q[6];
cu1(1.5707963267948966) q[5], q[1];
cu1(1.5707963267948966) q[0], q[5];
crz(1.5707963267948966) q[1], q[2];
crz(1.5707963267948966) q[7], q[2];
crz(1.5707963267948966) q[4], q[3];
crz(1.5707963267948966) q[6], q[7];
cu1(1.5707963267948966) q[6], q[0];
crz(1.5707963267948966) q[4], q[0];
cu1(1.5707963267948966) q[5], q[3];
cu1(1.5707963267948966) q[2], q[1];
cu1(1.5707963267948966) q[2], q[3];
crz(1.5707963267948966) q[6], q[7];
cz q[3], q[7];
cz q[6], q[0];
cu1(1.5707963267948966) q[7], q[8];
cz q[8], q[2];
cu1(1.5707963267948966) q[3], q[7];
crz(1.5707963267948966) q[6], q[2];
crz(1.5707963267948966) q[4], q[1];
crz(1.5707963267948966) q[8], q[5];
cu1(1.5707963267948966) q[1], q[0];
cu1(1.5707963267948966) q[3], q[6];
cu1(1.5707963267948966) q[0], q[1];
cu1(1.5707963267948966) q[6], q[5];
cz q[7], q[4];
cz q[6], q[7];
crz(1.5707963267948966) q[8], q[0];
crz(1.5707963267948966) q[7], q[3];
crz(1.5707963267948966) q[8], q[5];
cz q[6], q[3];
cz q[4], q[6];
crz(1.5707963267948966) q[3], q[8];
cz q[1], q[2];
cz q[8], q[5];
crz(1.5707963267948966) q[1], q[2];
cu1(1.5707963267948966) q[1], q[7];
cz q[0], q[8];
cz q[0], q[1];
cu1(1.5707963267948966) q[1], q[5];
cz q[6], q[1];
cz q[6], q[5];
crz(1.5707963267948966) q[6], q[3];
cz q[6], q[3];
cz q[7], q[8];
cz q[4], q[5];
cu1(1.5707963267948966) q[8], q[7];
crz(1.5707963267948966) q[4], q[2];
crz(1.5707963267948966) q[1], q[5];
cz q[3], q[8];
crz(1.5707963267948966) q[6], q[7];
crz(1.5707963267948966) q[0], q[3];
cz q[3], q[6];
cu1(1.5707963267948966) q[4], q[0];
cu1(1.5707963267948966) q[0], q[6];
cu1(1.5707963267948966) q[3], q[5];
cu1(1.5707963267948966) q[4], q[5];
cu1(1.5707963267948966) q[8], q[2];
cz q[3], q[6];
crz(1.5707963267948966) q[6], q[8];
cz q[1], q[4];
crz(1.5707963267948966) q[2], q[6];
cu1(1.5707963267948966) q[7], q[8];
cz q[5], q[0];
crz(1.5707963267948966) q[5], q[4];
crz(1.5707963267948966) q[4], q[5];
crz(1.5707963267948966) q[7], q[0];
cu1(1.5707963267948966) q[3], q[4];
cz q[7], q[1];
cu1(1.5707963267948966) q[5], q[0];
cu1(1.5707963267948966) q[2], q[4];
cu1(1.5707963267948966) q[7], q[8];
cu1(1.5707963267948966) q[1], q[2];
cu1(1.5707963267948966) q[6], q[3];
crz(1.5707963267948966) q[6], q[0];
cu1(1.5707963267948966) q[1], q[3];
cu1(1.5707963267948966) q[5], q[7];
cz q[7], q[6];
crz(1.5707963267948966) q[3], q[5];
crz(1.5707963267948966) q[6], q[4];
cu1(1.5707963267948966) q[1], q[0];
crz(1.5707963267948966) q[6], q[7];
crz(1.5707963267948966) q[0], q[5];
cu1(1.5707963267948966) q[3], q[0];
cu1(1.5707963267948966) q[8], q[6];
cz q[0], q[4];
cz q[1], q[8];
cz q[7], q[4];
cu1(1.5707963267948966) q[4], q[0];
