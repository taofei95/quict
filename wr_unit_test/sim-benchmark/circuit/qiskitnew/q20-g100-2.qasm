OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[8];
id q[17];
rxx(0) q[18], q[6];
swap q[14], q[17];
cy q[16], q[8];
id q[6];
sdg q[1];
rz(1.5707963267948966) q[1];
cy q[9], q[1];
id q[2];
ry(1.5707963267948966) q[11];
cu3(1.5707963267948966, 0, 0) q[8], q[9];
cy q[3], q[12];
cu3(1.5707963267948966, 0, 0) q[1], q[11];
rx(1.5707963267948966) q[14];
s q[16];
u1(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[9];
id q[11];
u2(1.5707963267948966, 1.5707963267948966) q[18];
swap q[4], q[13];
rz(1.5707963267948966) q[6];
t q[19];
tdg q[11];
sdg q[3];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[19];
s q[18];
h q[13];
t q[18];
t q[15];
h q[5];
s q[13];
t q[1];
rz(1.5707963267948966) q[1];
p(0) q[18];
u3(0, 0, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[12];
id q[19];
tdg q[10];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[13];
rz(1.5707963267948966) q[16];
p(0) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[14];
s q[14];
h q[17];
rx(1.5707963267948966) q[14];
s q[15];
cu3(1.5707963267948966, 0, 0) q[1], q[6];
s q[11];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[18];
id q[4];
p(0) q[12];
swap q[15], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[17];
cu3(1.5707963267948966, 0, 0) q[2], q[13];
tdg q[18];
s q[1];
u3(0, 0, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[1];
p(0) q[2];
id q[17];
u1(1.5707963267948966) q[14];
s q[3];
rzz(1.5707963267948966) q[10], q[3];
tdg q[15];
sdg q[9];
rx(1.5707963267948966) q[10];
cx q[10], q[5];
t q[12];
sdg q[2];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[9];
sdg q[9];
sdg q[3];
s q[9];
ry(1.5707963267948966) q[6];
t q[4];
rx(1.5707963267948966) q[3];
tdg q[14];
p(0) q[2];
t q[0];
h q[1];
rz(1.5707963267948966) q[14];
t q[7];
rz(1.5707963267948966) q[17];
id q[14];
cy q[9], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[9];
cy q[13], q[19];
cz q[12], q[9];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];