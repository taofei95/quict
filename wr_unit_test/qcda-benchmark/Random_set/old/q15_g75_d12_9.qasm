OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
s q[14];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[5];
rzz(1.5707963267948966) q[7], q[8];
rxx(0) q[2], q[11];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[13];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[12];
h q[3];
s q[1];
rz(1.5707963267948966) q[14];
cx q[12], q[8];
u2(1.5707963267948966, 1.5707963267948966) q[10];
ryy(1.5707963267948966) q[13], q[14];
t q[12];
cz q[5], q[4];
h q[9];
rx(1.5707963267948966) q[11];
h q[8];
rz(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[1], q[7];
x q[7];
s q[4];
s q[3];
tdg q[6];
rx(1.5707963267948966) q[2];
cx q[1], q[9];
rx(1.5707963267948966) q[3];
h q[2];
h q[8];
cy q[1], q[3];
cu3(1.5707963267948966, 0, 0) q[9], q[2];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[12];
rzz(1.5707963267948966) q[10], q[7];
tdg q[3];
p(0) q[2];
rx(1.5707963267948966) q[13];
rzz(1.5707963267948966) q[14], q[13];
s q[1];
rzz(1.5707963267948966) q[3], q[5];
s q[10];
t q[4];
u3(0, 0, 1.5707963267948966) q[7];
s q[4];
p(0) q[3];
rz(1.5707963267948966) q[14];
t q[3];
u3(0, 0, 1.5707963267948966) q[13];
ch q[5], q[10];
h q[14];
id q[13];
ch q[10], q[9];
u1(1.5707963267948966) q[6];
s q[1];
sdg q[8];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
s q[10];
rx(1.5707963267948966) q[9];
x q[3];
tdg q[11];
rz(1.5707963267948966) q[2];
s q[6];
ch q[2], q[9];
cy q[12], q[0];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[13];
ry(1.5707963267948966) q[10];
sdg q[6];