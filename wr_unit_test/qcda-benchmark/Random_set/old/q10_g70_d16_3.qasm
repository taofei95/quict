OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
s q[8];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[5];
t q[6];
crz(1.5707963267948966) q[5], q[7];
ryy(1.5707963267948966) q[5], q[1];
cu3(1.5707963267948966, 0, 0) q[6], q[5];
sdg q[5];
cu1(1.5707963267948966) q[1], q[3];
id q[3];
s q[0];
s q[0];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
p(0) q[6];
h q[4];
cx q[1], q[9];
h q[9];
h q[6];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[8];
rxx(0) q[5], q[3];
id q[5];
p(0) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
id q[2];
rxx(0) q[3], q[4];
tdg q[6];
crz(1.5707963267948966) q[7], q[0];
p(0) q[0];
rx(1.5707963267948966) q[9];
id q[6];
p(0) q[9];
tdg q[8];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
p(0) q[1];
s q[0];
ry(1.5707963267948966) q[1];
s q[6];
rxx(0) q[9], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[6];
ryy(1.5707963267948966) q[8], q[1];
ry(1.5707963267948966) q[4];
cz q[9], q[1];
ry(1.5707963267948966) q[3];
s q[7];
h q[0];
x q[5];
s q[3];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
s q[5];
sdg q[1];
rz(1.5707963267948966) q[9];
cz q[8], q[6];
cu3(1.5707963267948966, 0, 0) q[8], q[5];
u3(0, 0, 1.5707963267948966) q[8];
rxx(0) q[8], q[0];
cy q[5], q[3];
u3(0, 0, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[7];
cy q[5], q[4];