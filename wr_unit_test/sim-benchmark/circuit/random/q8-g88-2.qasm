OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
id q[3];
id q[2];
s q[3];
t q[7];
cx q[4], q[3];
tdg q[1];
s q[6];
id q[3];
x q[1];
x q[0];
s q[0];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[6];
crz(1.5707963267948966) q[0], q[1];
tdg q[4];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
tdg q[3];
s q[7];
cu3(1.5707963267948966, 0, 0) q[6], q[2];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[7];
x q[5];
rxx(0) q[2], q[7];
t q[1];
x q[1];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[7];
id q[2];
x q[5];
ry(1.5707963267948966) q[0];
sdg q[7];
rx(1.5707963267948966) q[3];
cz q[2], q[0];
h q[1];
s q[4];
cu1(1.5707963267948966) q[3], q[7];
tdg q[3];
rx(1.5707963267948966) q[5];
tdg q[2];
x q[4];
cz q[0], q[5];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[0];
t q[3];
h q[5];
rz(1.5707963267948966) q[5];
s q[2];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cu1(1.5707963267948966) q[6], q[3];
cu1(1.5707963267948966) q[6], q[3];
tdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[6];
tdg q[5];
rz(1.5707963267948966) q[2];
sdg q[0];
h q[0];
id q[1];
rx(1.5707963267948966) q[4];
rxx(0) q[0], q[5];
id q[1];
h q[4];
u1(1.5707963267948966) q[2];
x q[7];
u3(0, 0, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
sdg q[1];
h q[3];
h q[0];
h q[4];
tdg q[6];
rxx(0) q[4], q[6];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[5];
u1(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[3], q[2];
p(0) q[2];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
cu1(1.5707963267948966) q[4], q[5];
cy q[6], q[0];