OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
ry(1.5707963267948966) q[3];
id q[0];
t q[1];
s q[0];
rx(1.5707963267948966) q[3];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
t q[2];
tdg q[1];
cz q[0], q[3];
u1(1.5707963267948966) q[0];
t q[1];
u3(0, 0, 1.5707963267948966) q[1];
swap q[0], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rzz(1.5707963267948966) q[1], q[2];
id q[1];
p(0) q[0];
rz(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[1], q[2];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
sdg q[3];
swap q[0], q[2];
id q[1];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[0];
p(0) q[0];
id q[0];
sdg q[2];
sdg q[1];
rxx(0) q[0], q[2];
u1(1.5707963267948966) q[0];