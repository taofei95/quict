OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
s q[0];
h q[1];
sdg q[0];
h q[0];
u3(0, 0, 1.5707963267948966) q[1];
h q[1];
rz(1.5707963267948966) q[1];
x q[1];
t q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[1];
u1(1.5707963267948966) q[0];
x q[1];
rz(1.5707963267948966) q[0];
t q[0];
h q[1];
