OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
swap q[4], q[0];
tdg q[2];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
tdg q[2];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[3];
rxx(0) q[3], q[2];
rz(1.5707963267948966) q[1];
cu1(1.5707963267948966) q[1], q[2];
cu1(1.5707963267948966) q[4], q[3];
tdg q[0];
sdg q[2];
s q[0];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
tdg q[0];
s q[2];
t q[4];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[0];
crz(1.5707963267948966) q[3], q[1];
cx q[3], q[2];
id q[2];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
id q[3];
tdg q[0];
h q[4];
p(0) q[1];
id q[1];
u3(0, 0, 1.5707963267948966) q[3];