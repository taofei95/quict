OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
rx(1.5707963267948966) q[6];
z q[2];
u1(1.5707963267948966) q[0];
y q[2];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[0];
t q[5];
y q[0];
h q[0];
tdg q[4];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
t q[4];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
s q[0];
z q[5];
sdg q[2];
s q[0];
u3(0, 0, 1.5707963267948966) q[5];
t q[6];
t q[6];
h q[4];
h q[1];
h q[0];
rz(1.5707963267948966) q[0];
h q[5];
tdg q[4];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
h q[4];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[3];
h q[6];
x q[6];
ry(1.5707963267948966) q[0];
y q[6];
rx(1.5707963267948966) q[5];
