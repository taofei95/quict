OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
x q[1];
p(0) q[2];
u3(0, 0, 1.5707963267948966) q[5];
h q[1];
s q[1];
sdg q[1];
cz q[1], q[0];
x q[4];
rxx(0) q[6], q[1];
rz(1.5707963267948966) q[2];
sdg q[1];
x q[2];
tdg q[2];
id q[5];
t q[4];
h q[4];
h q[4];
ry(1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[2];
t q[1];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[1];
s q[0];
ry(1.5707963267948966) q[3];
tdg q[2];
t q[2];
ry(1.5707963267948966) q[6];
tdg q[2];
u3(0, 0, 1.5707963267948966) q[5];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
sdg q[4];
ry(1.5707963267948966) q[1];
s q[1];
x q[0];
s q[2];
t q[5];
p(0) q[5];
p(0) q[0];
cu1(1.5707963267948966) q[3], q[1];
t q[3];
ry(1.5707963267948966) q[5];
ryy(1.5707963267948966) q[3], q[0];
s q[5];
tdg q[6];
rz(1.5707963267948966) q[3];
cx q[5], q[0];
id q[2];
u1(1.5707963267948966) q[6];
crz(1.5707963267948966) q[7], q[0];
u3(0, 0, 1.5707963267948966) q[6];
swap q[5], q[6];
ry(1.5707963267948966) q[4];
h q[1];
id q[1];
sdg q[0];
t q[4];
p(0) q[7];
tdg q[6];
sdg q[2];
h q[2];
s q[1];
h q[6];
cy q[5], q[7];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[6];
s q[7];
x q[4];
s q[0];
rzz(1.5707963267948966) q[5], q[7];
rz(1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
cu1(1.5707963267948966) q[6], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
swap q[7], q[4];
x q[4];
rz(1.5707963267948966) q[5];
t q[0];
rx(1.5707963267948966) q[1];
s q[4];
h q[6];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[7];
u1(1.5707963267948966) q[6];
cy q[1], q[3];
u1(1.5707963267948966) q[6];