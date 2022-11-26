OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rxx(0) q[4], q[1];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
swap q[6], q[4];
id q[0];
cy q[6], q[3];
t q[7];
sdg q[5];
u3(0, 0, 1.5707963267948966) q[2];
cz q[5], q[7];
p(0) q[3];
s q[4];
rzz(1.5707963267948966) q[7], q[6];
sdg q[6];
rzz(1.5707963267948966) q[1], q[4];
rzz(1.5707963267948966) q[3], q[6];
h q[2];
u3(0, 0, 1.5707963267948966) q[5];
id q[2];
h q[0];
rzz(1.5707963267948966) q[3], q[2];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
p(0) q[4];
s q[1];
s q[5];
rxx(0) q[5], q[2];
id q[3];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
p(0) q[5];
p(0) q[7];
rzz(1.5707963267948966) q[5], q[6];
sdg q[1];
id q[4];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
tdg q[1];
rx(1.5707963267948966) q[6];
h q[1];
p(0) q[4];
rx(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[5], q[3];
id q[4];
sdg q[7];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
t q[2];
s q[2];
p(0) q[0];
cz q[7], q[6];
id q[1];
rx(1.5707963267948966) q[5];
s q[5];
rz(1.5707963267948966) q[2];
sdg q[3];
cz q[0], q[6];
tdg q[6];
h q[4];
rz(1.5707963267948966) q[3];
t q[5];
u3(0, 0, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[7], q[6];
h q[7];
id q[3];
p(0) q[5];
cx q[6], q[4];
rzz(1.5707963267948966) q[4], q[1];
id q[6];
cy q[6], q[2];
p(0) q[2];
id q[1];
t q[3];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
p(0) q[4];
p(0) q[7];
cz q[5], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[7];
tdg q[0];
id q[4];
u3(0, 0, 1.5707963267948966) q[0];
cx q[0], q[4];
rz(1.5707963267948966) q[4];