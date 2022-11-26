OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
s q[5];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[6], q[3];
tdg q[5];
cu3(1.5707963267948966, 0, 0) q[6], q[2];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cx q[0], q[4];
id q[0];
h q[5];
rz(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[0];
id q[6];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
h q[2];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
s q[0];
rz(1.5707963267948966) q[3];
p(0) q[7];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[2];
cx q[5], q[6];
h q[4];
id q[2];
sdg q[7];
h q[3];
h q[5];
rz(1.5707963267948966) q[5];
h q[3];
u3(0, 0, 1.5707963267948966) q[0];
t q[1];
cx q[4], q[6];
rx(1.5707963267948966) q[6];
id q[5];
id q[3];
tdg q[3];
id q[3];
u3(0, 0, 1.5707963267948966) q[4];
cx q[3], q[1];
u1(1.5707963267948966) q[7];
tdg q[5];
ry(1.5707963267948966) q[7];
rxx(0) q[6], q[0];
h q[1];
u1(1.5707963267948966) q[0];
sdg q[5];
id q[4];
h q[3];
tdg q[0];