OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rx(1.5707963267948966) q[5];
sdg q[5];
rz(1.5707963267948966) q[4];
tdg q[4];
cx q[2], q[4];
cu1(1.5707963267948966) q[3], q[0];
rz(1.5707963267948966) q[3];
id q[3];
cz q[2], q[4];
h q[2];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[2];
s q[3];
sdg q[1];
sdg q[0];
ry(1.5707963267948966) q[4];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[3];
tdg q[2];
s q[1];
rz(1.5707963267948966) q[4];
cz q[3], q[4];
id q[5];
sdg q[2];
t q[3];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
s q[2];
id q[5];
t q[3];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[4];
s q[4];
ry(1.5707963267948966) q[0];
id q[1];
tdg q[1];
cu1(1.5707963267948966) q[5], q[4];
h q[1];
p(0) q[5];
rz(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
id q[1];
rz(1.5707963267948966) q[5];
cz q[1], q[2];
crz(1.5707963267948966) q[1], q[3];
rz(1.5707963267948966) q[3];
id q[3];
cy q[4], q[2];
crz(1.5707963267948966) q[4], q[5];
cz q[2], q[5];
h q[4];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[4];
u1(1.5707963267948966) q[1];
sdg q[2];
t q[3];
rx(1.5707963267948966) q[1];
s q[0];
s q[3];
rxx(0) q[4], q[1];
rz(1.5707963267948966) q[0];
h q[2];
rxx(0) q[1], q[4];
u3(0, 0, 1.5707963267948966) q[3];
h q[1];
cy q[3], q[1];
ch q[0], q[3];
cx q[1], q[3];
t q[0];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[4];