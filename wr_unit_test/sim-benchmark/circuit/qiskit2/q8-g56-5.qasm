OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
u3(0, 0, 1.5707963267948966) q[0];
id q[7];
u3(0, 0, 1.5707963267948966) q[2];
cu1(1.5707963267948966) q[3], q[5];
sdg q[7];
u1(1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
t q[2];
id q[1];
cx q[3], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[4];
tdg q[0];
id q[6];
u3(0, 0, 1.5707963267948966) q[0];
cy q[2], q[4];
h q[6];
rzz(1.5707963267948966) q[1], q[2];
ry(1.5707963267948966) q[4];
cu1(1.5707963267948966) q[4], q[5];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[0];
t q[2];
rz(1.5707963267948966) q[1];
cz q[6], q[4];
cy q[4], q[1];
cu1(1.5707963267948966) q[3], q[1];
swap q[1], q[5];
u1(1.5707963267948966) q[4];
s q[3];
rx(1.5707963267948966) q[5];
sdg q[4];
h q[5];
p(0) q[1];
ry(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[3], q[4];
rz(1.5707963267948966) q[5];
h q[3];
rxx(0) q[6], q[1];
tdg q[7];
p(0) q[2];
swap q[4], q[6];
swap q[4], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[7];
cz q[6], q[4];
rxx(0) q[3], q[1];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
cy q[2], q[4];
rx(1.5707963267948966) q[7];
cu1(1.5707963267948966) q[7], q[3];
rzz(1.5707963267948966) q[4], q[3];
cy q[2], q[4];
t q[5];