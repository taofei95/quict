OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
sdg q[0];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[2];
p(0) q[3];
sdg q[2];
cu3(1.5707963267948966, 0, 0) q[3], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[2];
rxx(0) q[2], q[3];
ry(1.5707963267948966) q[3];
s q[2];
p(0) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rzz(1.5707963267948966) q[7], q[5];
p(0) q[4];
id q[6];
id q[1];
t q[3];
cu3(1.5707963267948966, 0, 0) q[6], q[0];
swap q[6], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cz q[7], q[6];
u3(0, 0, 1.5707963267948966) q[0];
cx q[0], q[3];
p(0) q[3];
sdg q[6];
rzz(1.5707963267948966) q[2], q[6];
h q[6];
sdg q[3];
sdg q[0];
tdg q[5];
tdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[5];
h q[3];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
t q[3];
t q[2];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[4];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[3];
h q[4];
tdg q[0];
cu1(1.5707963267948966) q[1], q[6];
u1(1.5707963267948966) q[5];
cy q[4], q[7];
s q[5];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[6];
cx q[0], q[1];
h q[1];
s q[7];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[6];
h q[5];
sdg q[6];
cu3(1.5707963267948966, 0, 0) q[0], q[3];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[5];
s q[4];
rxx(0) q[7], q[1];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];