OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
sdg q[1];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[4];
id q[5];
tdg q[7];
rz(1.5707963267948966) q[7];
s q[2];
p(0) q[1];
tdg q[2];
rz(1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[4], q[2];
rz(1.5707963267948966) q[1];
h q[7];
u3(0, 0, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[5];
rzz(1.5707963267948966) q[6], q[1];
tdg q[4];
p(0) q[3];
rz(1.5707963267948966) q[4];
s q[7];
sdg q[5];
s q[3];
tdg q[1];
tdg q[4];
cy q[7], q[0];
s q[5];
ry(1.5707963267948966) q[3];
tdg q[1];
cy q[3], q[1];
rzz(1.5707963267948966) q[2], q[1];
ry(1.5707963267948966) q[0];
s q[5];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[2];
h q[1];
h q[3];
crz(1.5707963267948966) q[6], q[7];
id q[4];
rxx(0) q[6], q[7];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[4], q[7];
h q[1];
cx q[5], q[3];
t q[2];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cx q[0], q[5];
tdg q[1];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];