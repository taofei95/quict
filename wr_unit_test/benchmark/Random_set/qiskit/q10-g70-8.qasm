OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
tdg q[0];
cu1(1.5707963267948966) q[5], q[2];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
s q[5];
swap q[0], q[3];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
t q[8];
u1(1.5707963267948966) q[2];
t q[0];
p(0) q[5];
rx(1.5707963267948966) q[8];
h q[9];
cu1(1.5707963267948966) q[9], q[6];
u3(0, 0, 1.5707963267948966) q[9];
h q[8];
tdg q[3];
swap q[1], q[4];
rx(1.5707963267948966) q[4];
id q[0];
u3(0, 0, 1.5707963267948966) q[3];
s q[3];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[6];
u1(1.5707963267948966) q[3];
cx q[2], q[4];
t q[2];
rx(1.5707963267948966) q[7];
t q[0];
cz q[4], q[1];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[6];
id q[8];
t q[6];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
sdg q[5];
rzz(1.5707963267948966) q[5], q[7];
u1(1.5707963267948966) q[1];
crz(1.5707963267948966) q[7], q[3];
p(0) q[1];
cz q[8], q[9];
t q[8];
id q[3];
ch q[6], q[0];
rx(1.5707963267948966) q[0];
crz(1.5707963267948966) q[6], q[5];
sdg q[3];
t q[9];
u3(0, 0, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[6];
s q[0];
t q[2];
t q[3];
rxx(0) q[9], q[6];
u3(0, 0, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[8];
h q[9];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[5];
h q[0];