OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
h q[5];
ry(1.5707963267948966) q[7];
cx q[8], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
id q[0];
p(0) q[3];
ry(1.5707963267948966) q[4];
cx q[0], q[3];
p(0) q[0];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[9];
h q[3];
u3(0, 0, 1.5707963267948966) q[3];
h q[8];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
cy q[3], q[4];
sdg q[3];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[9];
rxx(0) q[3], q[0];
p(0) q[3];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[5];
cu1(1.5707963267948966) q[8], q[5];
rxx(0) q[7], q[3];
u3(0, 0, 1.5707963267948966) q[7];
tdg q[4];
tdg q[1];
t q[8];
p(0) q[0];
id q[2];
rxx(0) q[5], q[6];
p(0) q[8];
tdg q[5];
rz(1.5707963267948966) q[5];
p(0) q[0];
p(0) q[6];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[0];
sdg q[0];
rz(1.5707963267948966) q[7];
cx q[5], q[7];
cx q[7], q[8];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
u1(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[8];
tdg q[7];
tdg q[6];
rxx(0) q[5], q[0];
h q[9];
t q[8];
id q[6];
h q[8];
u1(1.5707963267948966) q[7];
cy q[4], q[5];
t q[1];
cz q[8], q[7];
tdg q[8];
t q[3];
t q[9];
id q[9];
cz q[2], q[9];
id q[7];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[7];
h q[8];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cy q[0], q[5];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[0];
tdg q[8];
p(0) q[3];
rx(1.5707963267948966) q[5];
t q[1];
ry(1.5707963267948966) q[3];
s q[2];
t q[8];
sdg q[9];
tdg q[9];
s q[3];
p(0) q[2];
u3(0, 0, 1.5707963267948966) q[5];
rxx(0) q[6], q[4];
t q[1];
tdg q[2];