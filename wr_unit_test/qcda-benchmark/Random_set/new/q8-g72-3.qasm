OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
id q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
crz(1.5707963267948966) q[6], q[5];
s q[6];
rzz(1.5707963267948966) q[5], q[7];
t q[7];
rz(1.5707963267948966) q[5];
sdg q[5];
u1(1.5707963267948966) q[0];
cz q[2], q[1];
x q[4];
crz(1.5707963267948966) q[3], q[4];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[0];
tdg q[3];
cu3(1.5707963267948966, 0, 0) q[7], q[4];
rz(1.5707963267948966) q[6];
h q[4];
sdg q[4];
ry(1.5707963267948966) q[7];
id q[2];
s q[4];
rz(1.5707963267948966) q[0];
t q[6];
cz q[2], q[6];
cu3(1.5707963267948966, 0, 0) q[4], q[5];
id q[6];
x q[3];
h q[3];
rzz(1.5707963267948966) q[3], q[6];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
cx q[7], q[0];
tdg q[3];
s q[3];
h q[6];
tdg q[5];
crz(1.5707963267948966) q[6], q[4];
s q[6];
u1(1.5707963267948966) q[6];
crz(1.5707963267948966) q[7], q[1];
x q[2];
p(0) q[1];
x q[2];
h q[2];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
rxx(0) q[6], q[1];
ry(1.5707963267948966) q[4];
cz q[7], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[6];
id q[2];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
rxx(0) q[0], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ryy(1.5707963267948966) q[0], q[6];
id q[1];
x q[5];
sdg q[2];
cu1(1.5707963267948966) q[1], q[5];