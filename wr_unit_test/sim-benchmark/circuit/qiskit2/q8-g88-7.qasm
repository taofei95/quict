OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
id q[0];
rxx(0) q[4], q[7];
rxx(0) q[5], q[4];
u3(0, 0, 1.5707963267948966) q[4];
t q[7];
sdg q[7];
cy q[4], q[3];
s q[6];
sdg q[0];
h q[0];
u3(0, 0, 1.5707963267948966) q[3];
id q[4];
id q[0];
rxx(0) q[6], q[0];
u1(1.5707963267948966) q[4];
cu1(1.5707963267948966) q[3], q[6];
cx q[2], q[1];
tdg q[7];
t q[7];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
swap q[4], q[0];
rzz(1.5707963267948966) q[5], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[2];
sdg q[0];
cx q[3], q[5];
rzz(1.5707963267948966) q[2], q[0];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
swap q[6], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
h q[3];
swap q[4], q[0];
t q[3];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[3];
h q[1];
id q[7];
rz(1.5707963267948966) q[5];
t q[2];
u1(1.5707963267948966) q[4];
s q[6];
cx q[4], q[6];
h q[3];
rx(1.5707963267948966) q[1];
rzz(1.5707963267948966) q[5], q[1];
sdg q[6];
rxx(0) q[5], q[7];
tdg q[4];
cz q[5], q[7];
rzz(1.5707963267948966) q[5], q[2];
h q[3];
p(0) q[7];
s q[1];
h q[1];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
tdg q[2];
cx q[6], q[1];
h q[5];
u1(1.5707963267948966) q[3];
cz q[4], q[7];
ry(1.5707963267948966) q[2];
cz q[1], q[0];
rx(1.5707963267948966) q[1];
h q[4];
u3(0, 0, 1.5707963267948966) q[3];
cx q[7], q[4];
h q[0];
u1(1.5707963267948966) q[7];
cz q[4], q[6];
id q[0];
p(0) q[2];
p(0) q[5];
sdg q[5];
rzz(1.5707963267948966) q[7], q[1];
swap q[7], q[4];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
tdg q[4];
t q[7];
ry(1.5707963267948966) q[1];