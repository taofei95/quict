OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
ry(1.5707963267948966) q[0];
y q[1];
s q[1];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
z q[1];
h q[1];
rz(1.5707963267948966) q[1];
