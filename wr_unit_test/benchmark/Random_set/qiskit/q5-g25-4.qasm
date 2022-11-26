OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
tdg q[1];
sdg q[2];
cy q[3], q[4];
id q[4];
cy q[4], q[2];
rz(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[2], q[4];
cu1(1.5707963267948966) q[4], q[3];
cu3(1.5707963267948966, 0, 0) q[3], q[2];
tdg q[1];
t q[4];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[1];
sdg q[3];
ry(1.5707963267948966) q[0];
t q[3];
rzz(1.5707963267948966) q[2], q[1];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
id q[2];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];