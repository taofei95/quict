OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
id q[2];
h q[0];
id q[4];
sdg q[4];
id q[4];
tdg q[6];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
h q[4];
ry(1.5707963267948966) q[2];
s q[6];
rz(1.5707963267948966) q[3];
s q[5];
id q[1];
ry(1.5707963267948966) q[1];
h q[4];
h q[1];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
s q[6];
rz(1.5707963267948966) q[6];
s q[1];
u3(0, 0, 1.5707963267948966) q[0];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ch q[2], q[5];
h q[5];
rxx(0) q[4], q[0];
sdg q[0];
id q[2];
s q[4];
rz(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[5];
rxx(0) q[1], q[0];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
tdg q[1];
cz q[5], q[3];
id q[5];
sdg q[1];
rz(1.5707963267948966) q[5];
tdg q[4];
rzz(1.5707963267948966) q[1], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[6];
sdg q[3];
tdg q[4];
id q[1];
tdg q[6];
p(0) q[4];
sdg q[5];
rzz(1.5707963267948966) q[3], q[2];