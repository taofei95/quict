OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
swap q[0], q[1];
rx(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[6];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[1];
h q[6];
u1(1.5707963267948966) q[4];
tdg q[0];
t q[0];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[0];
s q[4];
h q[1];
u3(0, 0, 1.5707963267948966) q[6];
swap q[1], q[6];
tdg q[6];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
tdg q[2];
sdg q[2];
p(0) q[6];
h q[1];
rzz(1.5707963267948966) q[3], q[6];
h q[1];
cx q[1], q[4];
cu3(1.5707963267948966, 0, 0) q[5], q[6];
id q[5];
swap q[6], q[1];
rz(1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[5], q[6];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[0];
s q[6];
p(0) q[6];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[5];
h q[4];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[6];
id q[6];
h q[6];
t q[3];
id q[4];
s q[1];
p(0) q[3];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
sdg q[6];
tdg q[3];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[1];
s q[3];
rz(1.5707963267948966) q[1];