OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
h q[4];
cy q[1], q[3];
sdg q[6];
crz(1.5707963267948966) q[7], q[4];
u1(1.5707963267948966) q[5];
rzz(1.5707963267948966) q[1], q[4];
sdg q[6];
cz q[2], q[3];
cx q[1], q[2];
id q[2];
rx(1.5707963267948966) q[0];
h q[2];
cx q[7], q[1];
rz(1.5707963267948966) q[0];
t q[3];
s q[1];
p(0) q[1];
s q[2];
h q[1];
h q[3];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cy q[2], q[0];
p(0) q[2];
rzz(1.5707963267948966) q[2], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
id q[4];
cu1(1.5707963267948966) q[0], q[5];
t q[6];
s q[0];
s q[1];
tdg q[3];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
h q[4];
id q[3];
t q[1];
h q[7];
sdg q[4];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
id q[1];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[3];
cx q[1], q[7];
id q[7];
tdg q[1];
s q[7];
s q[3];
tdg q[6];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[2];
id q[6];
h q[2];
u3(0, 0, 1.5707963267948966) q[3];
t q[0];
s q[5];
s q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[6];
cz q[6], q[7];
p(0) q[2];
id q[5];
cx q[1], q[4];
id q[3];
cz q[0], q[4];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
t q[6];
h q[1];