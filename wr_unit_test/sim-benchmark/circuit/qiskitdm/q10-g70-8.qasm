OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u1(1.5707963267948966) q[9];
id q[2];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[7];
s q[7];
cu1(1.5707963267948966) q[6], q[1];
p(0) q[4];
id q[3];
h q[6];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
id q[4];
id q[6];
u3(0, 0, 1.5707963267948966) q[0];
tdg q[8];
rx(1.5707963267948966) q[9];
p(0) q[5];
cx q[7], q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[0];
cy q[1], q[6];
t q[4];
u3(0, 0, 1.5707963267948966) q[3];
cz q[2], q[7];
u3(0, 0, 1.5707963267948966) q[9];
h q[4];
id q[1];
h q[1];
rzz(1.5707963267948966) q[2], q[3];
ry(1.5707963267948966) q[7];
tdg q[9];
tdg q[3];
ry(1.5707963267948966) q[5];
u1(1.5707963267948966) q[8];
s q[3];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[9];
cz q[3], q[9];
u3(0, 0, 1.5707963267948966) q[6];
s q[5];
cy q[7], q[4];
u3(0, 0, 1.5707963267948966) q[2];
id q[8];
h q[8];
t q[8];
cz q[0], q[9];
cu1(1.5707963267948966) q[4], q[9];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[7], q[1];
s q[1];
p(0) q[8];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[2];
s q[5];
cy q[2], q[9];
rz(1.5707963267948966) q[2];
id q[6];
cy q[1], q[9];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[8];
p(0) q[0];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[5];
cz q[7], q[4];
t q[7];
s q[3];
sdg q[8];