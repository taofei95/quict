OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
t q[2];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[3];
id q[3];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[1];
sdg q[2];
u1(1.5707963267948966) q[2];
cy q[1], q[2];
ry(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[0], q[3];
rxx(0) q[0], q[2];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[1];
rzz(1.5707963267948966) q[2], q[0];
id q[3];
u3(0, 0, 1.5707963267948966) q[1];
cy q[2], q[0];
id q[2];
tdg q[2];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[2];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
s q[3];
u1(1.5707963267948966) q[0];
h q[2];
h q[0];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[0];