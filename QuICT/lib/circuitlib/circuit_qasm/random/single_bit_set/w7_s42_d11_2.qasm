OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
u1(1.5707963267948966) q[3];
s q[1];
rz(1.5707963267948966) q[4];
t q[5];
y q[2];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
h q[4];
h q[0];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[5];
y q[0];
rx(1.5707963267948966) q[0];
t q[3];
tdg q[0];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[2];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
s q[0];
y q[5];
tdg q[1];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
tdg q[0];
rx(1.5707963267948966) q[2];
tdg q[0];
z q[3];
h q[1];
x q[6];
