OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
u1(1.5707963267948966) q[1];
rxx(0) q[1], q[3];
rx(1.5707963267948966) q[3];
t q[1];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[1], q[3];
cu1(1.5707963267948966) q[3], q[1];
swap q[3], q[2];
sdg q[0];
s q[0];
u1(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
p(0) q[3];
u1(1.5707963267948966) q[0];
id q[0];
cz q[1], q[0];
cu1(1.5707963267948966) q[2], q[1];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[0];
id q[1];
t q[2];
cy q[1], q[2];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
t q[1];
cz q[1], q[2];
s q[1];
sdg q[0];
cu1(1.5707963267948966) q[1], q[2];
sdg q[1];
id q[2];
sdg q[2];
u1(1.5707963267948966) q[0];
t q[1];
p(0) q[0];
u3(0, 0, 1.5707963267948966) q[0];
swap q[3], q[2];
rx(1.5707963267948966) q[3];
p(0) q[2];
sdg q[2];
rx(1.5707963267948966) q[2];
t q[1];
t q[2];
tdg q[1];
u1(1.5707963267948966) q[2];
t q[1];
u3(0, 0, 1.5707963267948966) q[3];