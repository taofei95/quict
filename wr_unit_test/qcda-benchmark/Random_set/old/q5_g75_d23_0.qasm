OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
ry(1.5707963267948966) q[2];
t q[2];
x q[0];
h q[3];
h q[0];
sdg q[4];
t q[0];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[4], q[1];
rx(1.5707963267948966) q[3];
tdg q[0];
rx(1.5707963267948966) q[1];
h q[4];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[0];
ryy(1.5707963267948966) q[0], q[3];
t q[2];
t q[2];
u1(1.5707963267948966) q[3];
id q[0];
id q[3];
t q[3];
tdg q[3];
u1(1.5707963267948966) q[2];
x q[4];
cx q[2], q[4];
u1(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[1], q[4];
s q[3];
u3(0, 0, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
sdg q[0];
id q[0];
sdg q[3];
t q[0];
cu1(1.5707963267948966) q[1], q[2];
u1(1.5707963267948966) q[3];
tdg q[4];
rx(1.5707963267948966) q[0];
sdg q[4];
ch q[3], q[1];
rz(1.5707963267948966) q[1];
t q[0];
u1(1.5707963267948966) q[3];
s q[4];
t q[2];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[2];
t q[3];
cx q[0], q[2];
u3(0, 0, 1.5707963267948966) q[4];
s q[4];
u1(1.5707963267948966) q[1];
id q[1];
rz(1.5707963267948966) q[0];
cx q[2], q[3];
t q[3];
rx(1.5707963267948966) q[0];
h q[2];
h q[3];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[4];
s q[4];
tdg q[0];
cx q[4], q[2];
h q[2];
h q[3];
sdg q[2];