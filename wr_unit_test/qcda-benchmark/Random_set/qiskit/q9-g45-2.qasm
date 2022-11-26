OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
rz(1.5707963267948966) q[8];
sdg q[0];
h q[7];
t q[4];
p(0) q[4];
id q[8];
id q[4];
s q[7];
ry(1.5707963267948966) q[5];
tdg q[3];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[7];
cz q[3], q[7];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[7];
h q[2];
tdg q[2];
p(0) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[1];
t q[8];
h q[0];
s q[6];
crz(1.5707963267948966) q[5], q[3];
t q[7];
p(0) q[0];
s q[6];
rxx(0) q[1], q[8];
cz q[6], q[0];
rxx(0) q[0], q[1];
ry(1.5707963267948966) q[5];
s q[1];
cy q[3], q[5];
u3(0, 0, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
t q[1];
sdg q[6];
t q[0];
ch q[6], q[3];
tdg q[4];
cu1(1.5707963267948966) q[2], q[0];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[3];