OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
swap q[4], q[2];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
s q[4];
id q[1];
s q[0];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[1];
cz q[1], q[6];
t q[6];
rz(1.5707963267948966) q[5];
p(0) q[0];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
t q[6];
sdg q[6];
rx(1.5707963267948966) q[5];
s q[0];
s q[1];
sdg q[3];
rx(1.5707963267948966) q[0];
p(0) q[2];
rxx(0) q[5], q[3];
h q[2];
cx q[5], q[2];
u1(1.5707963267948966) q[6];
s q[1];
ch q[5], q[0];
id q[6];
ch q[6], q[5];
cu1(1.5707963267948966) q[0], q[4];
tdg q[5];
t q[3];
t q[4];
rz(1.5707963267948966) q[1];
sdg q[6];
x q[1];
s q[3];
t q[5];
x q[5];
t q[0];
ry(1.5707963267948966) q[2];
tdg q[4];
s q[0];
id q[0];
p(0) q[0];
sdg q[2];
h q[1];
h q[0];
rz(1.5707963267948966) q[0];
t q[6];
t q[2];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[4];
s q[4];
h q[6];
sdg q[3];
h q[4];
p(0) q[5];
u3(0, 0, 1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[3], q[5];
x q[5];