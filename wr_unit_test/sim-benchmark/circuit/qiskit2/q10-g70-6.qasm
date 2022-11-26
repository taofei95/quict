OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
tdg q[1];
sdg q[5];
p(0) q[0];
p(0) q[4];
rzz(1.5707963267948966) q[7], q[2];
cy q[3], q[5];
h q[3];
p(0) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
swap q[9], q[0];
s q[4];
s q[8];
tdg q[6];
h q[7];
tdg q[6];
rx(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[3];
cx q[7], q[6];
id q[6];
sdg q[9];
id q[4];
h q[6];
u1(1.5707963267948966) q[9];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[5];
cu1(1.5707963267948966) q[8], q[0];
p(0) q[6];
rx(1.5707963267948966) q[8];
cz q[2], q[1];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
id q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[1];
swap q[9], q[1];
id q[1];
sdg q[3];
p(0) q[7];
id q[7];
cy q[5], q[4];
rx(1.5707963267948966) q[6];
p(0) q[1];
u3(0, 0, 1.5707963267948966) q[6];
p(0) q[9];
sdg q[2];
cy q[8], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[2];
cx q[4], q[2];
p(0) q[6];
cy q[0], q[4];
p(0) q[2];
p(0) q[9];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[7];
h q[9];
rz(1.5707963267948966) q[9];
p(0) q[9];
u3(0, 0, 1.5707963267948966) q[1];
id q[4];
s q[7];
sdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[5];