OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
p(0) q[5];
t q[6];
rxx(0) q[5], q[4];
id q[0];
t q[3];
swap q[3], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[5];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
tdg q[4];
u1(1.5707963267948966) q[4];
t q[1];
ry(1.5707963267948966) q[3];
id q[5];
sdg q[5];
cx q[6], q[2];
u1(1.5707963267948966) q[2];
tdg q[6];
p(0) q[0];
ry(1.5707963267948966) q[0];
cx q[3], q[6];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
cy q[5], q[0];
cu1(1.5707963267948966) q[1], q[4];
u1(1.5707963267948966) q[6];
cy q[3], q[6];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
sdg q[3];
s q[4];