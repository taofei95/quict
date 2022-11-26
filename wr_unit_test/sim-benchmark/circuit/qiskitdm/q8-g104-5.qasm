OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[6];
s q[3];
sdg q[2];
h q[3];
rz(1.5707963267948966) q[4];
cx q[5], q[4];
id q[1];
cz q[5], q[2];
s q[0];
t q[3];
rzz(1.5707963267948966) q[3], q[6];
rx(1.5707963267948966) q[4];
s q[7];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
p(0) q[0];
id q[5];
u1(1.5707963267948966) q[6];
p(0) q[5];
id q[0];
s q[0];
cz q[6], q[2];
t q[0];
u3(0, 0, 1.5707963267948966) q[3];
t q[1];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[0];
tdg q[5];
s q[5];
cy q[5], q[4];
cz q[4], q[1];
rz(1.5707963267948966) q[3];
s q[3];
s q[7];
swap q[4], q[2];
tdg q[0];
tdg q[6];
p(0) q[1];
t q[7];
s q[4];
rzz(1.5707963267948966) q[5], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
t q[0];
id q[3];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
h q[6];
h q[2];
h q[5];
s q[6];
ry(1.5707963267948966) q[2];
id q[0];
sdg q[7];
cu1(1.5707963267948966) q[3], q[0];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
cz q[0], q[6];
rz(1.5707963267948966) q[4];
swap q[6], q[4];
sdg q[4];
swap q[3], q[4];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
p(0) q[1];
u1(1.5707963267948966) q[3];
sdg q[3];
p(0) q[2];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[7];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
tdg q[0];
ry(1.5707963267948966) q[0];
p(0) q[6];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[5], q[6];
h q[4];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
t q[0];
cy q[4], q[0];
id q[6];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[3];
p(0) q[4];
rz(1.5707963267948966) q[4];
sdg q[5];
ry(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
h q[4];
rz(1.5707963267948966) q[6];
ry(1.5707963267948966) q[0];