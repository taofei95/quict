OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
crz(1.5707963267948966) q[4], q[5];
ry(1.5707963267948966) q[2];
h q[4];
u3(0, 0, 1.5707963267948966) q[5];
x q[5];
p(0) q[5];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[4];
s q[4];
sdg q[2];
ryy(1.5707963267948966) q[0], q[1];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[2];
tdg q[1];
u1(1.5707963267948966) q[3];
p(0) q[4];
t q[2];
rz(1.5707963267948966) q[3];
h q[2];
h q[3];
h q[1];
rx(1.5707963267948966) q[5];
t q[3];
x q[5];
rx(1.5707963267948966) q[5];
t q[5];
x q[5];
cx q[1], q[5];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
s q[2];
rx(1.5707963267948966) q[4];
p(0) q[5];
rz(1.5707963267948966) q[4];
sdg q[3];
rz(1.5707963267948966) q[0];
x q[3];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
x q[4];
rxx(0) q[0], q[5];
tdg q[2];
h q[0];
s q[2];
t q[3];
crz(1.5707963267948966) q[5], q[3];
cz q[3], q[0];
x q[4];
s q[0];
tdg q[2];