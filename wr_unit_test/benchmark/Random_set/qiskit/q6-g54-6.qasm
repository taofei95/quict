OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
s q[0];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
cy q[3], q[4];
rx(1.5707963267948966) q[0];
cy q[3], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
p(0) q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
p(0) q[4];
sdg q[4];
tdg q[1];
h q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
crz(1.5707963267948966) q[3], q[4];
t q[0];
u1(1.5707963267948966) q[1];
swap q[3], q[1];
u3(0, 0, 1.5707963267948966) q[3];
cx q[1], q[0];
id q[4];
h q[2];
t q[5];
cx q[0], q[5];
p(0) q[0];
cz q[1], q[5];
s q[0];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
id q[1];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
t q[2];
rz(1.5707963267948966) q[3];
s q[3];
rxx(0) q[5], q[0];
h q[0];
sdg q[5];
ry(1.5707963267948966) q[4];
sdg q[4];
sdg q[4];
s q[4];
s q[3];
rx(1.5707963267948966) q[3];
swap q[4], q[1];
h q[3];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[5];
cu3(1.5707963267948966, 0, 0) q[5], q[0];
rx(1.5707963267948966) q[0];