OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
cu1(1.5707963267948966) q[12], q[16];
crz(1.5707963267948966) q[2], q[16];
cu1(1.5707963267948966) q[1], q[0];
cz q[2], q[9];
cz q[9], q[10];
crz(1.5707963267948966) q[13], q[8];
cu1(1.5707963267948966) q[13], q[6];
cz q[12], q[16];
cz q[14], q[3];
crz(1.5707963267948966) q[5], q[10];
cz q[0], q[3];
cz q[12], q[6];
crz(1.5707963267948966) q[6], q[0];
cz q[4], q[0];
crz(1.5707963267948966) q[16], q[8];
cu1(1.5707963267948966) q[7], q[17];
cz q[15], q[4];
cu1(1.5707963267948966) q[10], q[4];
cz q[4], q[2];
crz(1.5707963267948966) q[11], q[2];
cz q[16], q[3];
cu1(1.5707963267948966) q[11], q[1];
cz q[17], q[11];
cz q[12], q[13];
cz q[1], q[12];
cz q[4], q[10];
cz q[6], q[0];
cz q[9], q[4];
cu1(1.5707963267948966) q[10], q[1];
crz(1.5707963267948966) q[8], q[3];
cu1(1.5707963267948966) q[8], q[9];
crz(1.5707963267948966) q[2], q[16];
cu1(1.5707963267948966) q[6], q[16];
cz q[13], q[12];
cz q[8], q[17];
cu1(1.5707963267948966) q[11], q[1];
cz q[6], q[1];
cz q[1], q[16];
cz q[1], q[11];
cz q[5], q[12];
cu1(1.5707963267948966) q[9], q[0];
crz(1.5707963267948966) q[9], q[5];
cu1(1.5707963267948966) q[16], q[9];
crz(1.5707963267948966) q[1], q[15];
crz(1.5707963267948966) q[8], q[5];
crz(1.5707963267948966) q[6], q[3];
cu1(1.5707963267948966) q[5], q[1];
cz q[10], q[3];
cu1(1.5707963267948966) q[16], q[17];
cz q[6], q[7];
cz q[13], q[14];
cz q[3], q[6];
crz(1.5707963267948966) q[11], q[4];
cz q[11], q[16];
cu1(1.5707963267948966) q[0], q[15];
cz q[2], q[4];
cu1(1.5707963267948966) q[0], q[17];
crz(1.5707963267948966) q[16], q[1];
cz q[6], q[1];
cu1(1.5707963267948966) q[0], q[12];
cu1(1.5707963267948966) q[13], q[5];
crz(1.5707963267948966) q[5], q[4];
crz(1.5707963267948966) q[2], q[15];
crz(1.5707963267948966) q[15], q[16];
cu1(1.5707963267948966) q[10], q[9];
crz(1.5707963267948966) q[8], q[4];
crz(1.5707963267948966) q[13], q[3];
cz q[5], q[2];
cz q[6], q[12];
cu1(1.5707963267948966) q[12], q[15];
crz(1.5707963267948966) q[10], q[6];
cz q[0], q[16];
crz(1.5707963267948966) q[5], q[16];
crz(1.5707963267948966) q[17], q[13];
cz q[15], q[0];
cz q[5], q[10];
cz q[7], q[10];
cu1(1.5707963267948966) q[3], q[11];
cz q[11], q[0];
crz(1.5707963267948966) q[2], q[13];
cu1(1.5707963267948966) q[8], q[3];
cu1(1.5707963267948966) q[4], q[7];
crz(1.5707963267948966) q[17], q[13];
cz q[6], q[9];
cu1(1.5707963267948966) q[3], q[17];
cu1(1.5707963267948966) q[0], q[4];
crz(1.5707963267948966) q[12], q[4];
crz(1.5707963267948966) q[5], q[6];
cz q[5], q[4];
cu1(1.5707963267948966) q[5], q[13];
