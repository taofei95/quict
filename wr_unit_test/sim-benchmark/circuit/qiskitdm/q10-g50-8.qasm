OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
tdg q[3];
id q[5];
rxx(0) q[7], q[8];
p(0) q[2];
id q[1];
s q[0];
t q[2];
cz q[0], q[1];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
h q[8];
u1(1.5707963267948966) q[9];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
s q[4];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rzz(1.5707963267948966) q[8], q[5];
h q[3];
rxx(0) q[7], q[6];
t q[8];
rz(1.5707963267948966) q[7];
p(0) q[4];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[5];
s q[3];
rz(1.5707963267948966) q[1];
p(0) q[0];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
id q[4];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cx q[5], q[7];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[0];
rz(1.5707963267948966) q[5];
h q[1];
p(0) q[0];
t q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
s q[8];
u2(1.5707963267948966, 1.5707963267948966) q[5];
id q[5];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];