OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
swap q[4], q[0];
x q[5];
s q[3];
p(0) q[6];
sdg q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
rx(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[5], q[1];
rx(1.5707963267948966) q[3];
crz(1.5707963267948966) q[2], q[4];
s q[0];
id q[3];
cu3(1.5707963267948966, 0, 0) q[2], q[3];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[5];
h q[2];
ry(1.5707963267948966) q[2];
h q[2];
u3(0, 0, 1.5707963267948966) q[2];
x q[1];
tdg q[2];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
tdg q[5];
ry(1.5707963267948966) q[1];
h q[6];
rxx(0) q[0], q[1];
p(0) q[3];
sdg q[4];
ry(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[3], q[0];
cx q[4], q[1];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];