OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
cz q[1], q[2];
cz q[2], q[0];
crz(1.5707963267948966) q[0], q[2];
cu1(1.5707963267948966) q[1], q[0];
cz q[2], q[1];
cu1(1.5707963267948966) q[1], q[2];
crz(1.5707963267948966) q[2], q[0];
cz q[2], q[0];
cz q[2], q[0];
cu1(1.5707963267948966) q[0], q[2];
crz(1.5707963267948966) q[1], q[0];
cu1(1.5707963267948966) q[2], q[1];
crz(1.5707963267948966) q[1], q[2];
cu1(1.5707963267948966) q[1], q[0];
crz(1.5707963267948966) q[0], q[2];
cz q[1], q[0];
cz q[1], q[0];
cu1(1.5707963267948966) q[1], q[2];
cu1(1.5707963267948966) q[2], q[1];
crz(1.5707963267948966) q[2], q[1];
crz(1.5707963267948966) q[2], q[1];
crz(1.5707963267948966) q[0], q[2];
cz q[0], q[2];
crz(1.5707963267948966) q[0], q[1];
cz q[2], q[0];
crz(1.5707963267948966) q[2], q[0];
crz(1.5707963267948966) q[2], q[0];
crz(1.5707963267948966) q[1], q[2];
crz(1.5707963267948966) q[0], q[2];
crz(1.5707963267948966) q[0], q[1];
