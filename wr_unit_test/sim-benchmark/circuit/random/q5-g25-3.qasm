OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
ry(1.5707963267948966) q[0];
id q[1];
cx q[3], q[4];
id q[0];
ryy(1.5707963267948966) q[4], q[2];
ry(1.5707963267948966) q[4];
tdg q[4];
sdg q[4];
cz q[2], q[4];
rz(1.5707963267948966) q[2];
sdg q[3];
rxx(0) q[1], q[4];
p(0) q[1];
u3(0, 0, 1.5707963267948966) q[2];
t q[2];
rx(1.5707963267948966) q[1];
t q[0];
t q[3];
s q[1];
p(0) q[2];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[1];
ch q[2], q[4];
h q[2];